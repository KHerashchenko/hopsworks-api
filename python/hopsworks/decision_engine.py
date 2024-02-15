import logging
import pickle
from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass

import os
import humps
import pandas as pd

from hopsworks import client
from hopsworks.core import opensearch_api, dataset_api, kafka_api
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
# tf.keras.backend.set_floatx('float64') # didnt solve the error

from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from hsml.transformer import Transformer

from hsfs import connection as hsfs_conn
from hsml import connection as hsml_conn


class DecisionEngine(ABC):
    def __init__(self, configs_dict):
        self._name = configs_dict['name']
        self._configs_dict = configs_dict
        self._catalog_df = None
        self._embedding_model = None

        # todo refine api handles calls
        self._fs = hsfs_conn().get_feature_store(configs_dict['feature_store'])
        self._mr = hsml_conn().get_model_registry()

        client.init("hopsworks")
        self._client = client.get_instance()
        self._opensearch_api = opensearch_api.OpenSearchApi(self._client._project_id, self._client._project_name)
        self._dataset_api = dataset_api.DatasetApi(self._client._project_id)
        self._kafka_api = kafka_api.KafkaApi(self._client._project_id, self._client._project_name)

    @classmethod
    def from_response_json(cls, json_dict, project_id, project_name):
        json_decamelized = humps.decamelize(json_dict)
        if "count" not in json_decamelized:
            return cls(
                **json_decamelized, project_id=project_id, project_name=project_name
            )
        elif json_decamelized["count"] == 0:
            return []
        else:
            return [
                cls(**decision_engine, project_id=project_id, project_name=project_name)
                for decision_engine in json_decamelized["items"]
            ]

    def update_from_response_json(self, json_dict):
        json_decamelized = humps.decamelize(json_dict)
        self.__init__(**json_decamelized)
        return self

    @abstractmethod
    def build_catalog(self):
        pass

    @abstractmethod
    def run_data_validation(self):
        pass

    @abstractmethod
    def build_models(self):
        pass

    @abstractmethod
    def build_vector_db(self):
        pass

    @abstractmethod
    def build_deployments(self):
        pass


class RecommendationDecisionEngine(DecisionEngine):
    def build_catalog(self):

        # Creating catalog FG
        catalog_config = self._configs_dict['catalog']
        retrieval_config = self._configs_dict['model_configuration']['retrieval_model']
        catalog_idx_config = CatalogIndexConfig(**retrieval_config['catalog_index_config'])

        fg = self._fs.get_or_create_feature_group(
            name=catalog_config['feature_group_name'],
            description='Catalog for the Decision Engine project',
            primary_key=catalog_config['primary_key'],
            online_enabled=True,
            version=1
        )

        self._catalog_df = pd.read_csv(catalog_config['file_path'])

        pk_column_to_str = self._catalog_df.columns[catalog_idx_config.item_id_index]
        self._catalog_df[pk_column_to_str] = self._catalog_df[pk_column_to_str].astype(str)

        # todo tensorflow errors if col is of type float64, expecting float32
        price_column_to_float32 = self._catalog_df.columns[catalog_idx_config.price_index]
        self._catalog_df[price_column_to_float32] = self._catalog_df[price_column_to_float32].astype("float32")

        fg.insert(self._catalog_df)

        fv = self._fs.get_or_create_feature_view(
            name=catalog_config['feature_group_name'],
            query=fg.select_all(),
            version=1
        )

        fv.create_training_data(write_options={"use_spark": True})

    def run_data_validation(self):
        pass

    def build_models(self):

        # Creating retrieval model
        retrieval_config = self._configs_dict['model_configuration']['retrieval_model']

        catalog_idx_config = CatalogIndexConfig(**retrieval_config['catalog_index_config'])
        emb_dim = retrieval_config['item_space_dim']

        item_id_list = self._catalog_df.iloc[:, catalog_idx_config.item_id_index].astype(str).unique().tolist()
        name_list = self._catalog_df.iloc[:, catalog_idx_config.name_index].astype(str).unique().tolist()
        category_list = self._catalog_df.iloc[:, catalog_idx_config.category_index].astype(str).unique().tolist()

        self._embedding_model = ItemCatalogEmbedding(item_id_list, name_list, category_list, catalog_idx_config, emb_dim)

        tf.saved_model.save(self._embedding_model, "embedding_model")

        embedding_model_input_schema = Schema(self._catalog_df)
        embedding_model_output_schema = Schema([{
            "name": "embedding",
            "type": "double",
            "shape": [emb_dim],
        }])

        embedding_model_schema = ModelSchema(
            input_schema=embedding_model_input_schema,
            output_schema=embedding_model_output_schema
        )
        embedding_example = self._catalog_df.sample().to_dict("records")

        mr_embedding_model = self._mr.tensorflow.create_model(
            name="embedding_model",
            description="Model that generates embeddings from items catalog features",
            input_example=embedding_example,
            model_schema=embedding_model_schema,
        )
        mr_embedding_model.save("embedding_model")

        # Creating ranking model placeholder
        file_name = 'ranking_model.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump({}, file)

        ranking_model = self._mr.python.create_model(name="ranking_model",
                                               description="Ranking model that scores item candidates")
        ranking_model.save(model_path='ranking_model.pkl')

    def build_vector_db(self):

        # Creating Opensearch index
        os_client = OpenSearch(**self._opensearch_api.get_default_py_config())
        catalog_config = self._configs_dict['catalog']
        retrieval_config = self._configs_dict['model_configuration']['retrieval_model']

        index_name = self._opensearch_api.get_project_index(catalog_config['feature_group_name'])
        index_exists = os_client.indices.exists(index_name)
        # dev:
        if index_exists:
            os_client.indices.delete(index_name)

        if not index_exists:
            logging.info(f"Opensearch index name {index_name} does not exist. Creating.")
            index_body = {
                "settings": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                },
                "mappings": {
                    "properties": {
                        "my_vector1": {
                            "type": "knn_vector",
                            "dimension": retrieval_config['item_space_dim'],
                            "method": {
                                "name": "hnsw",
                                "space_type": retrieval_config['opensearch_index']['space_type'],
                                "engine": retrieval_config['opensearch_index']['engine'],
                                "parameters": {
                                    "ef_construction": 256,
                                    "m": 48
                                }
                            }
                        }
                    }
                }
            }
            response = os_client.indices.create(index_name, body=index_body)

        model = self._mr.get_model(
            name="embedding_model",
            version=1,
        )
        model_path = model.download()

        model_schema = model.model_schema['input_schema']['columnar_schema']
        item_features = [feat['name'] for feat in model_schema]

        items_ds = tf.data.Dataset.from_tensor_slices({col: self._catalog_df[col] for col in self._catalog_df})

        item_embeddings = items_ds.batch(2048).map(lambda x: (x[catalog_config['primary_key'][0]], self._embedding_model(x)))

        actions = []

        for batch in item_embeddings:
            item_id_list, embedding_list = batch
            item_id_list = item_id_list.numpy().astype(int)
            embedding_list = embedding_list.numpy()

            for item_id, embedding in zip(item_id_list, embedding_list):
                actions.append({
                    "_index": index_name,
                    "_id": item_id,
                    "_source": {
                        "my_vector1": embedding,
                    }
                })
        print("Example item vectors to be bulked: ", actions[:10])
        bulk(os_client, actions)

    def build_deployments(self):
        # copy transformer file into Hopsworks File System
        # uploaded_file_path = self._dataset_api.upload("ranking_transformer.py", "Resources", overwrite=True)
        # transformer_script_path = os.path.join("/Projects", self._client._project_name, uploaded_file_path).replace('\\', '/')
        transformer_script_path = os.path.join("/Projects", self._client._project_name, "Resources", "ranking_model_transformer.py").replace('\\', '/')

        # copy predictor file into Hopsworks File System
        # uploaded_file_path = dataset_api.upload("ranking_predictor.py", "Resources", overwrite=True)
        # predictor_script_path = os.path.join("/Projects", self._client._project_name, uploaded_file_path).replace('\\', '/')

        predictor_script_path = os.path.join("/Projects", self._client._project_name, "Resources", "ranking_model_predictor.py").replace('\\', '/')

        ranking_deployment_name = "rankingdeployment"
        ranking_model = self._mr.get_model("ranking_model", version=1)

        # define transformer
        ranking_transformer=Transformer(
            script_file=transformer_script_path,
            resources={"num_instances": 1},
        )

        # deploy ranking model
        ranking_deployment = ranking_model.deploy(
            name=ranking_deployment_name,
            description="Deployment that search for item candidates and scores them based on customer metadata",
            script_file=predictor_script_path,
            resources={"num_instances": 1},
            transformer=ranking_transformer,
        )

        # Creating deployment for logObservation endpoint

        # copy redirector file into Hopsworks File System
        # uploaded_file_path = dataset_api.upload("logObservations_redirect.py", "Resources", overwrite=True)
        # predictor_script_path = os.path.join("/Projects", self._client._project_name, uploaded_file_path).replace('\\', '/')

        redirector_script_path = os.path.join("/Projects", self._client._project_name, "Resources", "logObservations_redirect.py").replace('\\', '/')

        SCHEMA_NAME = '_'.join([self._configs_dict['name'], "observations"])
        TOPIC_NAME = '_'.join([self._configs_dict['name'], "logObservations"])

        avro_schema = {
            "type": "record",
            "name": "observations",
            "fields": []
        }

        self._kafka_api.create_schema(SCHEMA_NAME, avro_schema)
        # dev:
        try:
            self._kafka_api._delete_topic(TOPIC_NAME)
        except Exception:
            pass
        my_topic = self._kafka_api.create_topic(TOPIC_NAME, SCHEMA_NAME, 1, replicas=1, partitions=1)

        model = self._mr.python.create_model("logObservations_redirect")
        model.save(redirector_script_path, keep_original_files=True)
        predictor_script_path = os.path.join(model.version_path, "logObservations_redirect.py")
        deployment = model.deploy('observationsdeployment', script_file=predictor_script_path)

@dataclass
class CatalogIndexConfig:
    item_id_index: int = 0
    name_index: int = 1
    price_index: int = 2
    category_index: int = 3


class ItemCatalogEmbedding(tf.keras.Model):

    def __init__(self, item_id_list: List[str], category_list: List[str], name_list: List[str],
                 catalog_idx_config: CatalogIndexConfig, emb_dim: int):
        super().__init__()

        if emb_dim <= 0:
            raise ValueError("emb_dim must be a positive integer")

        if catalog_idx_config is None:
            catalog_idx_config = CatalogIndexConfig()

        self.item_embedding = tf.keras.Sequential([
            StringLookup(
                vocabulary=item_id_list,
                mask_token=None
            ),
            tf.keras.layers.Embedding(
                # We add an additional embedding to account for unknown tokens.
                len(item_id_list) + 1,
                emb_dim
            )
        ])

        self.normalized_price = tf.keras.layers.Normalization(axis=None)

        self.category_tokenizer = tf.keras.layers.StringLookup(
            vocabulary=category_list, mask_token=None
        )
        self.category_list_len = len(category_list)

        self.name_tokenizer = tf.keras.layers.StringLookup(
            vocabulary=name_list, mask_token=None
        )
        self.name_list_len = len(name_list)

        self.fnn = tf.keras.Sequential([
            tf.keras.layers.Dense(emb_dim, activation="relu"),
            tf.keras.layers.Dense(emb_dim)
        ])

        self.catalog_idx_config = catalog_idx_config

    def call(self, inputs):
        # todo remove hardcode, needs modifying CatalogIndexConfig
        item_id = inputs['item_id']
        category = inputs['category']
        price = inputs['price']
        name = inputs['name']

        category_embedding = tf.one_hot(
            self.category_tokenizer(category),
            self.category_list_len
        )

        name_embedding = tf.one_hot(
            self.name_tokenizer(name),
            self.name_list_len
        )

        concatenated_inputs = tf.concat([
            self.item_embedding(item_id),
            tf.reshape(self.normalized_price(price), (-1, 1)),
            category_embedding,
            name_embedding,
        ], axis=1)

        outputs = self.fnn(concatenated_inputs)

        return outputs


class SearchDecisionEngine(DecisionEngine):
    def __init__(self, config):
        self.config = config

    def build_catalog(self):
        # Implement logic to create feature groups for search engine based on config
        pass

    def run_data_validation(self):
        pass

    def build_models(self):
        # Implement logic to create search engine models based on config
        pass

    def build_deployments(self):
        # Implement logic to deploy search engine models
        pass