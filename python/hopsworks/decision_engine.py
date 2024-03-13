import logging
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Text
from dataclasses import dataclass

import os
import humps
import numpy as np
import pandas as pd

from hopsworks import client
from hopsworks.core import opensearch_api, dataset_api, kafka_api, job_api
from hsfs.feature import Feature
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow.keras.layers import TextVectorization
import tensorflow_recommenders as tfrs

# tf.keras.backend.set_floatx('float64') # didnt solve the error

from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from hsml.transformer import Transformer

from hsfs import connection as hsfs_conn
from hsml import connection as hsml_conn


class DecisionEngine(ABC):
    def __init__(self, configs_dict):
        self._name = configs_dict["name"]
        self._configs_dict = configs_dict
        self._prefix = "de_" + self._name + "_"
        self._catalog_df = None
        self._retrieval_model = None
        self._redirect_model = None

        # todo refine api handles calls
        client.init("hopsworks")
        self._client = client.get_instance()
        self._opensearch_api = opensearch_api.OpenSearchApi(
            self._client._project_id, self._client._project_name
        )
        self._dataset_api = dataset_api.DatasetApi(self._client._project_id)
        self._kafka_api = kafka_api.KafkaApi(
            self._client._project_id, self._client._project_name
        )
        self._jobs_api = job_api.JobsApi(
            self._client._project_id, self._client._project_name
        )

        self._fs = hsfs_conn().get_feature_store(
            self._client._project_name + "_featurestore"
        )
        self._mr = hsml_conn().get_model_registry()

        self._kafka_schema_name = self._prefix + "events" + "_1"
        self._kafka_topic_name = "_".join(
            [self._client._project_name, self._configs_dict["name"], "events"]
        )

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
    def build_feature_store(self):
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
    def build_feature_store(self):
        # Creating product list FG
        catalog_config = self._configs_dict["product_list"]

        items_fg = self._fs.get_or_create_feature_group(
            name=self._prefix + catalog_config["feature_view_name"],
            description="Catalog for the Decision Engine project",
            primary_key=[catalog_config["primary_key"]],
            online_enabled=True,
            version=1,
        )

        item_features = [
            Feature(name=feat, type=val["type"])
            for feat, val in catalog_config["schema"].items()
        ]
        items_fg.save(features=item_features)

        # Creating items FV
        items_fv = self._fs.get_or_create_feature_view(
            name=self._prefix + catalog_config["feature_view_name"],
            query=items_fg.select_all(),
            version=1,
        )
        
        self._catalog_df = pd.read_csv(
            catalog_config["file_path"],
            parse_dates=[
                feat
                for feat, val in catalog_config["schema"].items()
                if val["type"] == "timestamp"
            ],
        )
        items_fg.insert(self._catalog_df[catalog_config["schema"].keys()])
        # items_fg.add_tag(name="decision_engine", value={"use_case": self._configs_dict['use_case'], "name": self._configs_dict['name']})

        # TODO tensorflow errors if col is of type float64, expecting float32
        # TODO where timestamp feature transformation should happen? (converting into unix format)
        for feat, val in catalog_config["schema"].items():
            if val["type"] == "float":
                self._catalog_df[feat] = self._catalog_df[feat].astype("float32")
            if "transformation" in val.keys() and val["transformation"] == "timestamp":
                self._catalog_df[feat] = self._catalog_df[feat].astype(np.int64) // 10**9
        self._catalog_df[catalog_config['primary_key']] = self._catalog_df[catalog_config['primary_key']].astype(str)
                        
        # Creating events FG
        events_fg = self._fs.get_or_create_feature_group(
            name=self._prefix + "events",
            description="Events stream for the Decision Engine project",
            primary_key=["event_id"],  # TODO autoincrement?
            online_enabled=True,
            version=1,
        )

        # initialize with all possible context features even if user dropped some of them in config
        events_features = [
            Feature(name="event_id", type="bigint"),
            Feature(name="session_id", type="string"),
            Feature(name="event_timestamp", type="timestamp"),
            Feature(name="item_id", type="string"),
            Feature(name="event_type", type="string"),
            Feature(name="event_value", type="double"), # e.g. 0 or 1 for click, price for purchase
            Feature(name="event_weight", type="double"), # event_value multiplier
            Feature(name="longitude", type="double"), # TODO does it make sense to normalise session into separate fg?
            Feature(name="latitude", type="double"),
            Feature(name="language", type="string"),
            Feature(name="useragent", type="string"),
        ]

        events_fg.save(features=events_features)
        
        # Creating events FV
        events_fv = self._fs.get_or_create_feature_view(
            name=self._prefix + "events",
            query=events_fg.select_all(),
            version=1,
        )
        # events_fv.add_tag(name="decision_engine", value={"use_case": self._configs_dict['use_case'], "name": self._configs_dict['name']})

        events_fv.create_training_data(write_options={"use_spark": True})
        td_version, _ = events_fv.create_train_test_split(test_size=0.2, description='Models training dataset',
                                                            write_options={"wait_for_job": True})
        
        # Creating decisions FG
        decisions_fg = self._fs.get_or_create_feature_group(
            name=self._prefix + "decisions",
            description="Decisions logging for the Decision Engine project",
            primary_key=["decision_id"],  # TODO autoincrement?
            online_enabled=True,
            version=1,
        )

        decisions_features = [
            Feature(name="decision_id", type="bigint"),
            Feature(name="session_id", type="string"),
            Feature(
                name="session_activity", type=f"ARRAY <{catalog_config['schema'][catalog_config['primary_key']]['type']}>"
            ),  # item ids that user interacted with (all event types)
            Feature(
                name="predicted_items", type=f"ARRAY <{catalog_config['schema'][catalog_config['primary_key']]['type']}>"
            ),  # item ids received by getDecision
        ]

        decisions_fg.save(features=decisions_features)

    def build_models(self):
        # Creating retrieval model
        catalog_config = self._configs_dict["product_list"]
        retrieval_config = self._configs_dict["model_configuration"]["retrieval_model"]

        pk_index_list = (
            self._catalog_df[self._configs_dict["product_list"]["primary_key"]]
            .unique()
            .tolist()
        )
        categories_lists = {}
        text_features = {}
        for feat, val in catalog_config["schema"].items():
            if "transformation" not in val.keys():
                continue
            if val["transformation"] == "category":
                categories_lists[feat] = (
                    self._catalog_df[feat].astype(str).unique().tolist()
                )
            elif val["transformation"] == "text":
                text_features[feat] = self._catalog_df[feat].tolist()

        self._candidate_model = ItemCatalogEmbedding(
            self._configs_dict, pk_index_list, categories_lists
        )

        for feat, val in catalog_config["schema"].items():
            if "transformation" not in val.keys():
                continue
            if val["transformation"] in ["numeric", "timestamp"]:
                self._candidate_model.normalized_feats[feat].adapt(
                    self._catalog_df[feat].tolist()
                )            
            elif val["transformation"] == "text":
                self._candidate_model.texts_embeddings[feat].layers[0].adapt(
                    self._catalog_df[feat].tolist()
                )

        tf.saved_model.save(self._candidate_model, "candidate_model")
        
        candidate_model_schema = ModelSchema(
            input_schema=Schema(self._catalog_df),
            output_schema=Schema(
            [
                {
                    "name": "embedding",
                    "type": "double",
                    "shape": [retrieval_config["item_space_dim"]],
                }
            ]
        ),
        )
        candidate_example = self._catalog_df.sample().to_dict("records")

        candidate_model = self._mr.tensorflow.create_model(
            name=self._prefix + "candidate_model",
            description="Model that generates embeddings from items catalog features",
            input_example=candidate_example,
            model_schema=candidate_model_schema,
        )
        candidate_model.save("candidate_model")

        self._query_model = SequenceEmbedding(
            pk_index_list, retrieval_config["item_space_dim"]
        )
        query_model_module = QueryModelModule(self._query_model)
        # Define the input specifications for the instances
        instances_spec = {
            'context_item_ids': tf.TensorSpec(shape=(None,), dtype=tf.string, name='context_item_ids'),
        }

        # Get the concrete function for the query_model's compute_emb function using the specified input signatures
        signatures = query_model_module.compute_emb.get_concrete_function(instances_spec)

        # Save the query_model along with the concrete function signatures
        tf.saved_model.save(
            query_model_module,    # The model to save
            "query_model",         # Path to save the model
            signatures=signatures, # Concrete function signatures to include
        )

        query_model_schema = ModelSchema(
            input_schema=Schema(self._catalog_df.head()[self._configs_dict["product_list"]["primary_key"]]),
            output_schema=Schema(
            [
                {
                    "name": "embedding",
                    "type": "double",
                    "shape": [retrieval_config["item_space_dim"]],
                }
            ]
        ),
        )

        query_model = self._mr.tensorflow.create_model(
            name=self._prefix + "query_model",
            description="Model that generates embeddings from session interaction sequence",
            model_schema=query_model_schema,
        )
        query_model.save("query_model")

        # Creating ranking model
        self._ranking_model = RankingModel(self._candidate_model) # TODO add adapt() for features in a retrain job
        tf.saved_model.save(self._ranking_model, "ranking_model")
        
        ranking_model = self._mr.tensorflow.create_model(
            name=self._prefix + "ranking_model",
            description="Ranking model that scores item candidates",
        )
        ranking_model.save("ranking_model")
        # ranking_model.add_tag(name="decision_engine", value={"use_case": self._configs_dict['use_case'], "name": self._configs_dict['name']})

        # Creating Redirect model for events redirect to Kafka
        self._redirect_model = self._mr.python.create_model(
            self._prefix + "events_redirect",
            description="Workaround model for redirecting events into Kafka",
        )
        redirector_script_path = os.path.join(
            "/Projects",
            self._client._project_name,
            "Resources",
            "decision-engine",
            "events_redirect_predictor.py",
        )
        self._redirect_model.save(redirector_script_path, keep_original_files=True)
        # ranking_model.add_tag(name="decision_engine", value={"use_case": self._configs_dict['use_case'], "name": self._configs_dict['name']})

    def build_vector_db(self):
        # Creating Opensearch index
        os_client = OpenSearch(**self._opensearch_api.get_default_py_config())
        catalog_config = self._configs_dict["product_list"]
        retrieval_config = self._configs_dict["model_configuration"]["retrieval_model"]

        index_name = self._opensearch_api.get_project_index(
            catalog_config["feature_view_name"]
        )
        index_exists = os_client.indices.exists(index_name)
        # dev:
        if index_exists:
            os_client.indices.delete(index_name)
            index_exists = False

        if not index_exists:
            logging.info(
                f"Opensearch index name {index_name} does not exist. Creating."
            )
            index_body = {
                "settings": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                },
                "mappings": {
                    "properties": {
                        self._prefix
                        + "vector": {
                            "type": "knn_vector",
                            "dimension": retrieval_config["item_space_dim"],
                            "method": {
                                "name": "hnsw",
                                "space_type": "innerproduct",
                                "engine": "faiss",
                                "parameters": {"ef_construction": 256, "m": 48},
                            },
                        }
                    }
                },
            }
            response = os_client.indices.create(index_name, body=index_body)

        items_ds = tf.data.Dataset.from_tensor_slices(
            {col: self._catalog_df[col] for col in self._catalog_df}
        )

        item_embeddings = items_ds.batch(2048).map(
            lambda x: (x[catalog_config["primary_key"]], self._candidate_model(x))
        )

        actions = []

        for batch in item_embeddings:
            item_id_list, embedding_list = batch
            item_id_list = item_id_list.numpy().astype(int)
            embedding_list = embedding_list.numpy()

            for item_id, embedding in zip(item_id_list, embedding_list):
                actions.append(
                    {
                        "_index": index_name,
                        "_id": item_id,
                        "_source": {
                            self._prefix + "vector": embedding,
                        },
                    }
                )
        logging.info(f"Example item vectors to be bulked: {actions[:10]}")
        bulk(os_client, actions)

    def build_deployments(self):
        # Creating deployment for ranking model
        mr_ranking_model = self._mr.get_model(
            name=self._prefix + "ranking_model", version=1
        )

        transformer_script_path = os.path.join(
            "/Projects",
            self._client._project_name,
            "Resources",
            "decision-engine",
            "ranking_model_transformer.py",
        )
        ranking_transformer = Transformer(
            script_file=transformer_script_path, resources={"num_instances": 1}
        )

        ranking_deployment = mr_ranking_model.deploy(
            name=(self._prefix + "ranking_deployment").replace("_", "").lower(),
            description="Deployment that searches for item candidates and scores them based on session context and query embedding",
            resources={"num_instances": 1},
            transformer=ranking_transformer,
        )
        
        mr_query_model = self._mr.get_model(
            name=self._prefix + "query_model", version=1
        )
        
        transformer_script_path = os.path.join(
            "/Projects",
            self._client._project_name,
            "Resources",
            "decision-engine",
            "query_model_transformer.py",
        )
        query_transformer = Transformer(
            script_file=transformer_script_path, resources={"num_instances": 1}
        )

        query_deployment = mr_query_model.deploy(
            name=(self._prefix + "query_deployment").replace("_", "").lower(),
            description="Deployment that computes query embedding from session activity",
            resources={"num_instances": 1},
            transformer=query_transformer,
        )

        # Creating deployment for events endpoint
        redirector_script_path = os.path.join(
            self._redirect_model.version_path, "events_redirect_predictor.py"
        )
        deployment = self._redirect_model.deploy(
            (self._prefix + "events_redirect_deployment")
            .replace("_", "")
            .lower(),
            script_file=redirector_script_path,
        )
        
        # dev:
        try:
            self._kafka_api._delete_topic(self._kafka_topic_name)
        except Exception:
            pass
        my_topic = self._kafka_api.create_topic(
            self._kafka_topic_name, self._kafka_schema_name, 1, replicas=1, partitions=1
        )

    def build_jobs(self):
        # The job retraining the models. 
        py_config = self._jobs_api.get_configuration("PYTHON")
        py_config["appPath"] = os.path.join(
            "/Projects",
            self._client._project_name,
            "Resources",
            "decision-engine",
            "retrain_job.py",
        )
        py_config["defaultArgs"] = f"-name {self._name}"
        job = self._jobs_api.create_job(
            self._prefix + "retrain_job", py_config
        )

        # The job consuming events from Kafka topic.
        spark_config = self._jobs_api.get_configuration("PYSPARK")
        spark_config["appPath"] = os.path.join(
            "/Projects",
            self._client._project_name,
            "Resources",
            "decision-engine",
            "events_consume_job.py",
        )
        spark_config["defaultArgs"] = f"-name {self._name}"
        job = self._jobs_api.create_job(
            self._prefix + "events_consume_job", spark_config
        )
        
        # TODO create a job for retraining models if product list is updated


class ItemCatalogEmbedding(tf.keras.Model):
    """
    Candidate embedding tower of the Retrieval model
    """

    def __init__(
        self,
        configs_dict: dict,
        pk_index_list: List[str],
        categories_lists: Dict[str, List[str]],
    ):
        super().__init__()

        self._configs_dict = configs_dict
        item_space_dim = self._configs_dict["model_configuration"]["retrieval_model"]["item_space_dim"]

        self.pk_embedding = tf.keras.Sequential(
            [
                StringLookup(vocabulary=pk_index_list, mask_token=None),
                tf.keras.layers.Embedding(
                    # We add an additional embedding to account for unknown tokens.
                    len(pk_index_list) + 1,
                    item_space_dim,
                ),
            ]
        )

        self.categories_tokenizers = {}
        self.categories_lens = {}
        for feat, lst in categories_lists.items():
            self.categories_tokenizers[feat] = tf.keras.layers.StringLookup(
                vocabulary=lst, mask_token=None
            )
            self.categories_lens[feat] = len(lst)

        vocab_size = 1000
        self.texts_embeddings = {}
        self.normalized_feats = {}
        for feat, val in self._configs_dict["product_list"]["schema"].items():
            if "transformation" not in val.keys():
                continue
            if val["transformation"] == "text":
                self.texts_embeddings[feat] = tf.keras.Sequential(
                    [
                        TextVectorization(
                            max_tokens=vocab_size,
                        ),
                        tf.keras.layers.Embedding(
                            vocab_size, item_space_dim, mask_zero=True
                        ),
                        tf.keras.layers.GlobalAveragePooling1D(),
                    ]
                )
            elif val["transformation"] in ["numeric", "timestamp"]: # TODO change feature engineering for timestamps cause this is fucked
                self.normalized_feats[feat] = tf.keras.layers.Normalization(axis=None) 

        self.fnn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(item_space_dim, activation="relu"),
                tf.keras.layers.Dense(item_space_dim),
            ]
        )

    def call(self, inputs):
        # Explicitly name input tensors
        pk_inputs = inputs[self._configs_dict['product_list']["primary_key"]]
        category_inputs = {feat: inputs[feat] for feat in self.categories_tokenizers}
        text_inputs = {feat: inputs[feat] for feat in self.texts_embeddings}
        numeric_inputs = {feat: inputs[feat] for feat in self.normalized_feats}

        layers = [self.pk_embedding(pk_inputs)]

        for feat, val in self._configs_dict["product_list"]["schema"].items():
            if "transformation" not in val.keys():
                continue
            if val["transformation"] == "category":
                layers.append(
                    tf.one_hot(
                        self.categories_tokenizers[feat](category_inputs[feat]),
                        self.categories_lens[feat],
                    )
                )
            elif val["transformation"] == "text":
                layers.append(self.texts_embeddings[feat](text_inputs[feat]))
            elif val["transformation"] in ["numeric", "timestamp"]:
                layers.append(
                    tf.reshape(self.normalized_feats[feat](numeric_inputs[feat]), (-1, 1))
                )

        concatenated_inputs = tf.concat(layers, axis=1)
        outputs = self.fnn(concatenated_inputs)

        return outputs



class SequenceEmbedding(tf.keras.Model):
    def __init__(self, pk_index_list, item_space_dim):
        super().__init__()
        self.string_lookup = tf.keras.layers.StringLookup(
            vocabulary=pk_index_list, mask_token=None
        )
        self.embedding = tf.keras.layers.Embedding(
            len(pk_index_list) + 1, item_space_dim
        )
        self.gru = tf.keras.layers.GRU(item_space_dim, return_sequences=False)

    def call(self, inputs):
        x = self.string_lookup(inputs)
        x = self.embedding(x)
        # Reshape x to add a timesteps dimension if it's not inherently present
        # This assumes that the sequence length is 1 for each input
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=1)
        x = self.gru(x)
        return x


class QueryModelModule(tf.Module):
    def __init__(self, query_model):
        self.query_model = query_model

    @tf.function()
    def compute_emb(self, instances):
        # Compute the query embeddings
        query_emb = self.query_model(instances["context_item_ids"])
        # Ensure the output is a dictionary of tensors
        return {
            "query_emb": query_emb,
        }


class SessionModel(tf.keras.Model):
    """
    Session embedding model used in the Ranking model.
    """

    def __init__(self, candidate_model):
        super().__init__()
        self._candidate_model = candidate_model

        self.latitude = tf.keras.layers.Normalization(axis=None)
        self.longitude = tf.keras.layers.Normalization(axis=None)

        language_codes = ['en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'sv'] # TODO provide full list
        self.language = tf.keras.layers.StringLookup(
            vocabulary=language_codes, mask_token=None
        )
        self.language_len = len(language_codes)

        vocab_size = 100
        text_embed_size = 16
        self.useragent = tf.keras.Sequential(
            [
                TextVectorization(
                    max_tokens=vocab_size,
                ),
                tf.keras.layers.Embedding(vocab_size, text_embed_size, mask_zero=True),
                tf.keras.layers.GlobalAveragePooling1D(),
            ]
        )

        self._available_feature_transformations = {
            "longitude": self.longitude,
            "latitude": self.latitude,
            "language": self.language,
            "useragent": self.useragent,
        }
        
        # Compute predictions.
        self.ratings = tf.keras.Sequential(
            [
                # Learn multiple dense layers.
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                # Make rating predictions in the final layer.
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, inputs):
        item_features = inputs["item_features"]
        session_features = inputs["session_features"]

        candidate_embedding = self._candidate_model(item_features)

        session_embedding = []
        for feature in session_features:
            if feature in self._available_feature_transformations:
                session_embedding.append(
                    self._available_feature_transformations[feature](
                        session_features[feature]
                    )
                )

        return self.ratings(
            tf.concat(session_embedding + [candidate_embedding], axis=1)
        )


class RankingModel(tfrs.models.Model):
    """
    Ranking model.
    """

    def __init__(self, candidate_model):
        super().__init__()
        self._session_model = SessionModel(candidate_model)
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    def call(self, inputs):
        return self._session_model(inputs)

    def compute_loss(self, inputs, training=False):
        labels = inputs.pop("score")

        rating_predictions = self(inputs)

        # The task computes the loss and the metrics.
        return self.task(labels=labels, predictions=rating_predictions)


class SearchDecisionEngine(DecisionEngine):
    def __init__(self, config):
        self.config = config

    def build_feature_store(self):
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
