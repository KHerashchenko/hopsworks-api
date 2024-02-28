import logging
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass

import os
import humps
import numpy
import pandas as pd

from hopsworks import client
from hopsworks.core import opensearch_api, dataset_api, kafka_api, job_api
from hsfs.feature import Feature
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow.keras.layers import TextVectorization
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
        self._prefix = 'de_' + self._name + '_'
        self._catalog_df = None
        self._embedding_model = None
        self._redirect_model = None

        # todo refine api handles calls
        client.init("hopsworks")
        self._client = client.get_instance()
        self._opensearch_api = opensearch_api.OpenSearchApi(self._client._project_id, self._client._project_name)
        self._dataset_api = dataset_api.DatasetApi(self._client._project_id)
        self._kafka_api = kafka_api.KafkaApi(self._client._project_id, self._client._project_name)
        self._jobs_api = job_api.JobsApi(self._client._project_id, self._client._project_name)

        self._fs = hsfs_conn().get_feature_store(self._client._project_name + "_featurestore")
        self._mr = hsml_conn().get_model_registry()

        self._kafka_schema_name = '_'.join([self._client._project_name, self._configs_dict['name'], "observations"])
        self._kafka_topic_name = '_'.join([self._client._project_name, self._configs_dict['name'], "logObservations"])

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
    def build_models(self):
        pass

    @abstractmethod
    def build_vector_db(self):
        pass

    @abstractmethod
    def build_deployments(self):
        pass

    @abstractmethod
    def build_jobs(self):
        pass

    @property
    def prefix(self):
        """Prefix of DE engine entities"""
        return self._prefix

    @property
    def configs(self):
        """Configs dict of DE project"""
        return self._configs_dict

    @property
    def name(self):
        """Name of DE project"""
        return self._name

    @property
    def kafka_topic_name(self):
        """Name of Kafka topic used by DE project for observations"""
        return self._kafka_topic_name


class RecommendationDecisionEngine(DecisionEngine):
    def build_catalog(self):

        # Creating catalog FG
        catalog_config = self._configs_dict['product_list']

        fg = self._fs.get_or_create_feature_group(
            name=self._prefix + catalog_config['feature_view_name'],
            description='Catalog for the Decision Engine project',
            primary_key=[catalog_config['primary_key']],
            online_enabled=True,
            version=1
        )

        item_features = [Feature(name=feat, type=val['type']) for feat, val in catalog_config['schema'].items()]
        fg.save(features=item_features)

        self._catalog_df = pd.read_csv(catalog_config['file_path'],
                                       parse_dates=[feat for feat, val in catalog_config['schema'].items() if
                                                    val['type'] == 'timestamp'])
        fg.insert(self._catalog_df[catalog_config['schema'].keys()])
        # fv.add_tag(name="decision_engine", value={"use_case": self._configs_dict['use_case'], "name": self._configs_dict['name']})

        # todo tensorflow errors if col is of type float64, expecting float32
        for feat, val in catalog_config['schema'].items():
            if val['type'] == 'float':
                self._catalog_df[feat] = self._catalog_df[feat].astype("float32")

        fv = self._fs.get_or_create_feature_view(
            name=self._prefix + catalog_config['feature_view_name'],
            query=fg.select_all(),
            version=1
        )
        # fv.add_tag(name="decision_engine", value={"use_case": self._configs_dict['use_case'], "name": self._configs_dict['name']})

        fv.create_training_data(write_options={"use_spark": True})

    def build_models(self):

        # Creating retrieval model
        catalog_config = self._configs_dict['product_list']
        retrieval_config = self._configs_dict['model_configuration']['retrieval_model']

        pk_index_list = self._catalog_df[self._configs_dict['product_list']['primary_key']].astype(str).unique().tolist()
        categories_lists = {}
        text_features = {}
        for feat, val in catalog_config['schema'].items():
            if 'transformation' not in val.keys():
                continue
            if val['transformation'] == 'category':
                categories_lists[feat] = self._catalog_df[feat].astype(str).unique().tolist()
            elif val['transformation'] == 'text':
                text_features[feat] = self._catalog_df[feat].tolist()

        self._embedding_model = ItemCatalogEmbedding(self._configs_dict, pk_index_list, categories_lists)

        for feat, val in catalog_config['schema'].items():
            if 'transformation' not in val.keys():
                continue
            if val['transformation'] == 'numeric':
                self._embedding_model.normalized_feats[feat].adapt(self._catalog_df[feat].tolist())
            elif val['transformation'] == 'text':
                self._embedding_model.texts_embeddings[feat].layers[0].adapt(self._catalog_df[feat].tolist())

        tf.saved_model.save(self._embedding_model, "embedding_model")

        embedding_model_input_schema = Schema(self._catalog_df)
        embedding_model_output_schema = Schema([{
            "name": "embedding",
            "type": "double",
            "shape": [retrieval_config['item_space_dim']],
        }])

        embedding_model_schema = ModelSchema(
            input_schema=embedding_model_input_schema,
            output_schema=embedding_model_output_schema
        )
        embedding_example = self._catalog_df.sample().to_dict("records")

        embedding_model = self._mr.tensorflow.create_model(
            name=self._prefix + "embedding_model",
            description="Model that generates embeddings from items catalog features",
            input_example=embedding_example,
            model_schema=embedding_model_schema,
        )
        embedding_model.save("embedding_model")
        # embedding_model.add_tag(name="decision_engine", value={"use_case": self._configs_dict['use_case'], "name": self._configs_dict['name']})

        # Creating ranking model placeholder
        file_name = 'ranking_model.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump({}, file)

        ranking_model = self._mr.python.create_model(name=self._prefix + "ranking_model",
                                                     description="Ranking model that scores item candidates")
        ranking_model.save(model_path='ranking_model.pkl')
        # ranking_model.add_tag(name="decision_engine", value={"use_case": self._configs_dict['use_case'], "name": self._configs_dict['name']})

        # Creating logObservations model for events redirect to Kafka
        self._redirect_model = self._mr.python.create_model(self._prefix + "logObservations_redirect",
                                                            description="Workaround model for redirecting observations into Kafka")
        redirector_script_path = os.path.join("/Projects", self._client._project_name, "Resources",
                                              "logObservations_redirect_predictor.py").replace('\\', '/')
        self._redirect_model.save(redirector_script_path, keep_original_files=True)

    def build_vector_db(self):

        # Creating Opensearch index
        os_client = OpenSearch(**self._opensearch_api.get_default_py_config())
        catalog_config = self._configs_dict['catalog']
        retrieval_config = self._configs_dict['model_configuration']['retrieval_model']

        index_name = self._opensearch_api.get_project_index(catalog_config['feature_view_name'])
        index_exists = os_client.indices.exists(index_name)
        # dev:
        if index_exists:
            os_client.indices.delete(index_name)
            index_exists = False

        if not index_exists:
            logging.info(f"Opensearch index name {index_name} does not exist. Creating.")
            index_body = {
                "settings": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                },
                "mappings": {
                    "properties": {
                        self._prefix + "vector": {
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

        items_ds = tf.data.Dataset.from_tensor_slices({col: self._catalog_df[col] for col in self._catalog_df})

        item_embeddings = items_ds.batch(2048).map(
            lambda x: (x[catalog_config['primary_key']], self._embedding_model(x)))

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
                        self._prefix + "vector": embedding,
                    }
                })
        logging.info(f"Example item vectors to be bulked: {actions[:10]}")
        bulk(os_client, actions)

    def build_deployments(self):
        # Creating deployment for ranking model
        mr_ranking_model = self._mr.get_model(name=self._prefix + "ranking_model", version=1)

        transformer_script_path = os.path.join("/Projects", self._client._project_name, "Resources",
                                               "ranking_model_transformer.py").replace('\\', '/')
        predictor_script_path = os.path.join("/Projects", self._client._project_name, "Resources",
                                             "ranking_model_predictor.py").replace('\\', '/')

        # define transformer
        ranking_transformer = Transformer(script_file=transformer_script_path, resources={"num_instances": 1})

        ranking_deployment = mr_ranking_model.deploy(
            name=(self._prefix + "ranking_deployment").replace("_", "").lower(),
            description="Deployment that search for item candidates and scores them based on customer metadata",
            script_file=predictor_script_path,
            resources={"num_instances": 1},
            transformer=ranking_transformer,
        )

        # Creating deployment for logObservation endpoint
        mr_redirect_model = self._mr.get_model(name=self._prefix + "logObservations_redirect", version=1)
        redirector_script_path = os.path.join(self._redirect_model.version_path,
                                              "logObservations_redirect_predictor.py")
        deployment = self._redirect_model.deploy(
            (self._prefix + 'logObservations_redirect_deployment').replace("_", "").lower(),
            script_file=redirector_script_path)

        # creating Kafka topic for logObservation endpoint
        avro_schema = {
            "type": "record",
            "name": "observations",
            "fields": []
        }

        self._kafka_api.create_schema(self._kafka_schema_name, avro_schema)
        # dev:
        try:
            self._kafka_api._delete_topic(self._kafka_topic_name)
        except Exception:
            pass
        my_topic = self._kafka_api.create_topic(self._kafka_topic_name, self._kafka_schema_name, 1, replicas=1,
                                                partitions=1)

    def build_jobs(self):

        # The job retraining the ranking model. Compares the size of current training dataset and "observations" FG.
        # If diff > 10%, creates new training dataset, retrains ranking model and updates deployment.
        spark_config = self._jobs_api.get_configuration("PYTHON")
        spark_config['appPath'] = "/Resources/ranking_model_retrain_job.py"
        job = self._jobs_api.create_job(self._prefix + "ranking_model_retrain_job", spark_config)

        # The job for consuming observations from Kafka topic. Runs on schedule, inserts stream into observations FG.
        # On the first run, autodetects event schema and creates "observations" FG, "training" FV and empty training dataset.
        spark_config = self._jobs_api.get_configuration("PYSPARK")
        spark_config['appPath'] = "/Resources/logObservations_consume_job.py"
        job = self._jobs_api.create_job(self._prefix + "logObservations_consume_job", spark_config)


class ItemCatalogEmbedding(tf.keras.Model):

    def __init__(self, configs_dict: dict, pk_index_list: List[str], categories_lists: Dict[str, List[str]]):
        super().__init__()

        self._configs_dict = configs_dict
        item_space_dim = self._configs_dict['model_configuration']['retrieval_model']['item_space_dim']

        self.pk_embedding = tf.keras.Sequential([
            StringLookup(
                vocabulary=pk_index_list,
                mask_token=None
            ),
            tf.keras.layers.Embedding(
                # We add an additional embedding to account for unknown tokens.
                len(pk_index_list) + 1,
                item_space_dim
            )
        ])

        self.categories_tokenizers = {}
        self.categories_lens = {}
        for feat, lst in categories_lists.values():
            self.categories_tokenizers[feat] = tf.keras.layers.StringLookup(vocabulary=lst, mask_token=None)
            self.categories_lens[feat] = len(lst)

        vocab_size = 1000
        self.texts_embeddings = []
        self.normalized_feats = []
        for feat, val in self._configs_dict['product_list']['schema'].items():
            if 'transformation' not in val.keys():
                continue
            if val['transformation'] == 'text':
                self.texts_embeddings[feat] = tf.keras.Sequential([
                    TextVectorization(
                        max_tokens=vocab_size,
                        output_mode="int",
                        output_sequence_length=100,
                    ),
                    tf.keras.layers.Embedding(
                        vocab_size,
                        item_space_dim,
                    )
                ])
            elif val['transformation'] == 'numeric':
                self.normalized_feats[feat] = tf.keras.layers.Normalization(axis=None)


        self.fnn = tf.keras.Sequential([
            tf.keras.layers.Dense(item_space_dim, activation="relu"),
            tf.keras.layers.Dense(item_space_dim)
        ])

    def call(self, inputs):

        layers = [
            self.pk_embedding(inputs[self._configs_dict['primary_key']])
        ]

        for feat, val in self._configs_dict['product_list']['schema'].items():
            if 'transformation' not in val.keys():
                continue
            if val['transformation'] == 'category':
                layers.append(tf.one_hot(self.categories_tokenizers[feat](inputs[feat]), self.categories_lens[feat]))
            elif val['transformation'] == 'text':
                layers.append(self.texts_embeddings[feat](inputs[feat]))
            elif val['transformation'] == 'numeric':
                layers.append(tf.reshape(self.normalized_feats[feat](inputs[feat]), (-1, 1)))

        concatenated_inputs = tf.concat(layers, axis=1)
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
