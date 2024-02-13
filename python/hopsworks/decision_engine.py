import logging
import pickle
from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass

import humps
import pandas as pd

from hopsworks import client
from hopsworks.core import opensearch_api
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

from hsfs import connection as hsfs_conn
from hsml import connection as hsml_conn


class DecisionEngine(ABC):
    def __init__(self, configs_dict):
        self._name = configs_dict['name']
        self._configs_dict = configs_dict
        self._catalog_df = None

        # todo refine api handles calls
        self._fs = hsfs_conn().get_feature_store(configs_dict['feature_store'])
        self._mr = hsml_conn().get_model_registry()

        client.init("hopsworks")
        self._client = client.get_instance()
        self._os_api = opensearch_api.OpenSearchApi(self._client._project_id, self._client._project_name)

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

        fg = self._fs.get_or_create_feature_group(
            name=catalog_config['feature_group_name'],
            description='Catalog for the Decision Engine project',
            version=1,
            primary_key=catalog_config['primary_key'],
            online_enabled=True,
        )

        self._catalog_df = pd.read_csv(catalog_config['file_path'])
        fg.insert(self._catalog_df)

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

        item_model = ItemCatalogEmbedding(item_id_list, name_list, category_list, catalog_idx_config, emb_dim)

        tf.saved_model.save(item_model, "embedding_model")

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
        os_client = OpenSearch(**self._os_api.get_default_py_config())
        catalog_config = self._configs_dict['catalog']
        retrieval_config = self._configs_dict['model_configuration']['retrieval_model']

        index_name = self._os_api.get_project_index(catalog_config['feature_group_name'])
        index_exists = os_client.indices.exists(index_name)

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
        item_model = tf.saved_model.load(model_path)

        model_schema = model.model_schema['input_schema']['columnar_schema']
        item_features = [feat['name'] for feat in model_schema]

        items_ds = tf.data.Dataset.from_tensor_slices({col: self._catalog_df[col] for col in self._catalog_df})

        item_embeddings = items_ds.batch(2048).map(lambda x: (x[catalog_config['primary_key'][0]], item_model(x)))

        actions = []

        for batch in item_embeddings:
            item_id_list, embedding_list = batch
            item_id_list = item_id_list.numpy().astype(int)
            embedding_list = embedding_list.numpy()

            for item_id, embedding in zip(item_id_list, embedding_list):
                actions.append({
                    "_index": index_name,
                    "_id": str(item_id),
                    "_source": {
                        "my_vector": embedding,
                    }
                })

        bulk(os_client, actions)


    def build_deployments(self):
        pass


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
        self.fnn = tf.keras.Sequential([
            tf.keras.layers.Dense(emb_dim, activation="relu"),
            tf.keras.layers.Dense(emb_dim)
        ])

        self.catalog_idx_config = catalog_idx_config

    def call(self, inputs):
        item_id = inputs[:, self.catalog_idx_config.item_id_index]
        category = inputs[:, self.catalog_idx_config.category_index]
        price = inputs[:, self.catalog_idx_config.price_index]

        category_embedding = tf.one_hot(
            self.category_tokenizer(category),
            self.category_list_len
        )

        concatenated_inputs = tf.concat([
            self.item_embedding(item_id),
            tf.reshape(self.normalized_price(price), (-1, 1)),
            category_embedding,
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