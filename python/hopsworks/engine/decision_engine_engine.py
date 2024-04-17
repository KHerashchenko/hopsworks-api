import logging
from abc import ABC, abstractmethod

import os
import numpy as np
import pandas as pd

from hsfs.feature import Feature
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

from hopsworks.engine import decision_engine_model
import tensorflow as tf

# tf.keras.backend.set_floatx('float64') # didnt solve the error

from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from hsml.transformer import Transformer


class DecisionEngineEngine(ABC):
    def __init__(self):
        pass

    def setup_decision_engine(self, de):
        self.build_feature_store(de)
        self.build_models(de)
        self.build_vector_db(de)
        self.build_deployments(de)
        self.build_jobs(de)

    @abstractmethod
    def build_feature_store(self, de):
        pass

    @abstractmethod
    def build_models(self, de):
        pass

    @abstractmethod
    def build_vector_db(self, de):
        pass

    @abstractmethod
    def build_deployments(self, de):
        pass

    @abstractmethod
    def build_jobs(self, de):
        pass


class RecommendationDecisionEngineEngine(DecisionEngineEngine):
    def build_feature_store(self, de):
        # Creating product list FG
        catalog_config = de._configs_dict["product_list"]

        items_fg = de._fs.get_or_create_feature_group(
            name=de._prefix + catalog_config["feature_view_name"],
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
        items_fv = de._fs.get_or_create_feature_view(
            name=de._prefix + catalog_config["feature_view_name"],
            query=items_fg.select_all(),
            version=1,
        )

        downloaded_file_path = de._dataset_api.download(catalog_config["file_path"], overwrite=True)

        de._catalog_df = pd.read_csv(
            downloaded_file_path,
            parse_dates=[
                feat
                for feat, val in catalog_config["schema"].items()
                if val["type"] == "timestamp"
            ],
        )
        items_fg.insert(de._catalog_df[catalog_config["schema"].keys()])

        # TODO tensorflow errors if col is of type float64, expecting float32
        # TODO where timestamp feature transformation should happen? (converting into unix format)
        for feat, val in catalog_config["schema"].items():
            if val["type"] == "float":
                de._catalog_df[feat] = de._catalog_df[feat].astype("float32")
            if "transformation" in val.keys() and val["transformation"] == "timestamp":
                de._catalog_df[feat] = (
                    de._catalog_df[feat].astype(np.int64) // 10**9
                )
        de._catalog_df[catalog_config["primary_key"]] = de._catalog_df[
            catalog_config["primary_key"]
        ].astype(str)

        # Creating events FG
        events_fg = de._fs.get_or_create_feature_group(
            name=de._prefix + "events",
            description="Events stream for the Decision Engine project",
            primary_key=["event_id"],  # TODO autoincrement?
            online_enabled=True,
            stream=True,
            version=1,
        )

        # initialize with all possible context features even if user dropped some of them in config
        events_features = [
            Feature(name="event_id", type="bigint"),
            Feature(name="session_id", type="string"),
            Feature(name="event_timestamp", type="timestamp"),
            Feature(name="item_id", type="string"),
            Feature(name="event_type", type="string"),
            Feature(
                name="event_value", type="double"
            ),  # e.g. 0 or 1 for click, price for purchase
            Feature(name="event_weight", type="double"),  # event_value multiplier
            Feature(
                name="longitude", type="double"
            ),  # TODO does it make sense to normalise session into separate fg?
            Feature(name="latitude", type="double"),
            Feature(name="language", type="string"),
            Feature(name="useragent", type="string"),
        ]

        events_fg.save(features=events_features)

        # Creating events FV
        events_fv = de._fs.get_or_create_feature_view(
            name=de._prefix + "events",
            query=events_fg.select_all(),
            version=1,
        )

        events_fv.create_training_data(write_options={"use_spark": True})
        td_version, _ = events_fv.create_train_test_split(
            test_size=0.2,
            description="Models training dataset",
            write_options={"wait_for_job": True},
        )

        # Creating decisions FG
        decisions_fg = de._fs.get_or_create_feature_group(
            name=de._prefix + "decisions",
            description="Decisions logging for the Decision Engine project",
            primary_key=["decision_id", "session_id"], 
            online_enabled=True,
            version=1,
        )

        decisions_features = [
            Feature(name="decision_id", type="bigint"),
            Feature(name="session_id", type="string"),
            Feature(
                name="session_activity",
                type=f"ARRAY <{catalog_config['schema'][catalog_config['primary_key']]['type']}>",
            ),  # item ids that user interacted with (all event types)
            Feature(
                name="predicted_items",
                type=f"ARRAY <{catalog_config['schema'][catalog_config['primary_key']]['type']}>",
            ),  # item ids received by getDecision
        ]

        decisions_fg.save(features=decisions_features)

    def build_models(self, de):
        # Creating retrieval model
        catalog_config = de._configs_dict["product_list"]
        retrieval_config = de._configs_dict["model_configuration"]["retrieval_model"]

        pk_index_list = (
            de._catalog_df[de._configs_dict["product_list"]["primary_key"]]
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
                    de._catalog_df[feat].astype(str).unique().tolist()
                )
            elif val["transformation"] == "text":
                text_features[feat] = de._catalog_df[feat].tolist()

        de._candidate_model = decision_engine_model.ItemCatalogEmbedding(
            de._configs_dict, pk_index_list, categories_lists
        )

        for feat, val in catalog_config["schema"].items():
            if "transformation" not in val.keys():
                continue
            if val["transformation"] in ["numeric", "timestamp"]:
                de._candidate_model.normalized_feats[feat].adapt(
                    de._catalog_df[feat].tolist()
                )
            # elif val["transformation"] == "text":
            #     de._candidate_model.texts_embeddings[feat].layers[0].adapt(
            #         de._catalog_df[feat].tolist()
            #     )

        tf.saved_model.save(de._candidate_model, "candidate_model")

        candidate_model_schema = ModelSchema(
            input_schema=Schema(de._catalog_df),
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
        candidate_example = de._catalog_df.sample().to_dict("records")

        candidate_model = de._mr.tensorflow.create_model(
            name=de._prefix + "candidate_model",
            description="Model that generates embeddings from items catalog features",
            input_example=candidate_example,
            model_schema=candidate_model_schema,
        )
        candidate_model.save("candidate_model")

        de._query_model = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(vocabulary=pk_index_list, mask_token=None),
                tf.keras.layers.Embedding(
                    len(pk_index_list) + 1, retrieval_config["item_space_dim"]
                ),
                tf.keras.layers.GRU(retrieval_config["item_space_dim"]),
            ]
        )

        query_model_module = decision_engine_model.QueryModelModule(de._query_model)
        # Define the input specifications for the instances
        instances_spec = {
            "context_item_ids": tf.TensorSpec(
                shape=(None, None), dtype=tf.string, name="context_item_ids"
            ),
        }

        # Get the concrete function for the query_model's compute_emb function using the specified input signatures
        signatures = query_model_module.compute_emb.get_concrete_function(
            instances_spec
        )

        # Save the query_model along with the concrete function signatures
        tf.saved_model.save(
            query_model_module,
            "query_model",
            signatures=signatures,
        )

        query_model_schema = ModelSchema(
            input_schema=Schema(
                de._catalog_df.head()[
                    de._configs_dict["product_list"]["primary_key"]
                ]
            ),
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

        mr_query_model = de._mr.tensorflow.create_model(
            name=de._prefix + "query_model",
            description="Model that generates embeddings from session interaction sequence",
            model_schema=query_model_schema,
        )
        mr_query_model.save("query_model")

        # Creating ranking model
        de._ranking_model = decision_engine_model.RankingModel(
            de._configs_dict, categories_lists
        )

        for feat, val in catalog_config["schema"].items():
            if "transformation" not in val.keys():
                continue
            if val["transformation"] in ["numeric", "timestamp"]:
                de._ranking_model.normalized_feats[feat].adapt(
                    de._catalog_df[feat].tolist()
                )
            # elif val["transformation"] == "text":
            #     de._candidate_model.texts_embeddings[feat].layers[0].adapt(
            #         de._catalog_df[feat].tolist()
            #     )

        ranking_model_module = decision_engine_model.RankingModelModule(
            de._ranking_model
        )

        # TODO remove hardcode features from items df
        instances_spec = {
            "article_id": tf.TensorSpec(
                shape=(None,), dtype=tf.string, name="article_id"
            ),
            "detail_desc": tf.TensorSpec(
                shape=(None,), dtype=tf.string, name="detail_desc"
            ),
            "price": tf.TensorSpec(shape=(None,), dtype=tf.float32, name="price"),
            "prod_name": tf.TensorSpec(
                shape=(None,), dtype=tf.string, name="prod_name"
            ),
            "product_type_name": tf.TensorSpec(
                shape=(None,), dtype=tf.string, name="product_type_name"
            ),
            "t_dat": tf.TensorSpec(shape=(None,), dtype=tf.int64, name="t_dat"),
            "longitude": tf.TensorSpec(
                shape=(None,), dtype=tf.float32, name="longitude"
            ),
            "latitude": tf.TensorSpec(shape=(None,), dtype=tf.float32, name="latitude"),
            "language": tf.TensorSpec(shape=(None,), dtype=tf.string, name="language"),
            "useragent": tf.TensorSpec(
                shape=(None,), dtype=tf.string, name="useragent"
            ),
        }
        signatures = ranking_model_module.serve.get_concrete_function(instances_spec)
        tf.saved_model.save(
            ranking_model_module, "ranking_model", signatures=signatures
        )

        mr_ranking_model = de._mr.tensorflow.create_model(
            name=de._prefix + "ranking_model",
            description="Ranking model that scores item candidates",
        )
        mr_ranking_model.save("ranking_model")

        # Creating Redirect model for events redirect to Kafka
        de._redirect_model = de._mr.python.create_model(
            de._prefix + "events_redirect",
            description="Workaround model for redirecting events into Kafka",
        )
        redirector_script_path = os.path.join(
            "/Projects",
            de._client._project_name,
            "Resources",
            "decision-engine",
            "events_redirect_predictor.py",
        )
        de._redirect_model.save(redirector_script_path, keep_original_files=True)

    def build_vector_db(self, de):
        # Creating Opensearch index
        os_client = OpenSearch(**de._opensearch_api.get_default_py_config())
        catalog_config = de._configs_dict["product_list"]
        retrieval_config = de._configs_dict["model_configuration"]["retrieval_model"]

        index_name = de._opensearch_api.get_project_index(
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
                        de._prefix
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
            os_client.indices.create(index_name, body=index_body)

        items_ds = tf.data.Dataset.from_tensor_slices(
            {col: de._catalog_df[col] for col in de._catalog_df}
        )

        item_embeddings = items_ds.batch(2048).map(
            lambda x: (x[catalog_config["primary_key"]], de._candidate_model(x))
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
                            de._prefix + "vector": embedding,
                        },
                    }
                )
        logging.info(f"Example item vectors to be bulked: {actions[:10]}")
        bulk(os_client, actions)

    def build_deployments(self, de):
        # Creating deployment for ranking model
        mr_ranking_model = de._mr.get_model(
            name=de._prefix + "ranking_model", version=1
        )

        transformer_script_path = os.path.join(
            "/Projects",
            de._client._project_name,
            "Resources",
            "decision-engine",
            "ranking_model_transformer.py",
        )
        ranking_transformer = Transformer(
            script_file=transformer_script_path, resources={"num_instances": 1}
        )

        ranking_deployment = mr_ranking_model.deploy(
            name=(de._prefix + "ranking_deployment").replace("_", "").lower(),
            description="Deployment that searches for item candidates and scores them based on session context and query embedding",
            resources={"num_instances": 1},
            transformer=ranking_transformer,
        )

        mr_query_model = de._mr.get_model(
            name=de._prefix + "query_model", version=1
        )

        transformer_script_path = os.path.join(
            "/Projects",
            de._client._project_name,
            "Resources",
            "decision-engine",
            "query_model_transformer.py",
        )
        query_transformer = Transformer(
            script_file=transformer_script_path, resources={"num_instances": 1}
        )

        query_deployment = mr_query_model.deploy(
            name=(de._prefix + "query_deployment").replace("_", "").lower(),
            description="Deployment that computes query embedding from session activity",
            resources={"num_instances": 1},
            transformer=query_transformer,
        )

        # Creating deployment for events endpoint
        redirector_script_path = os.path.join(
            de._redirect_model.version_path, "events_redirect_predictor.py"
        )
        redirect_deployment = de._redirect_model.deploy(
            (de._prefix + "events_redirect_deployment").replace("_", "").lower(),
            script_file=redirector_script_path,
            resources={"num_instances": 1},
            description="Deployment that redirects session activity to Kafka",
        )

        try:
            de._kafka_api._delete_topic(de._kafka_topic_name)
        except Exception:
            pass
        
        my_topic = de._kafka_api.create_topic(
            de._kafka_topic_name, de._kafka_schema_name, 1, replicas=1, partitions=1
        )

    def build_jobs(self, de):
        # The job retraining the models.
        py_config = de._jobs_api.get_configuration("PYTHON")
        py_config["appPath"] = os.path.join(
            "/Projects",
            de._client._project_name,
            "Resources",
            "decision-engine",
            "retrain_job.py",
        )
        py_config["defaultArgs"] = f"-name {de._name}"
        job = de._jobs_api.create_job(de._prefix + "retrain_job", py_config)

        # The job consuming events from Kafka topic.
        spark_config = de._jobs_api.get_configuration("PYSPARK")
        spark_config["appPath"] = os.path.join(
            "/Projects",
            de._client._project_name,
            "Resources",
            "decision-engine",
            "events_consume_job.py",
        )
        spark_config["defaultArgs"] = f"-name {de._name}"
        job = de._jobs_api.create_job(
            de._prefix + "events_consume_job", spark_config
        )


class SearchDecisionEngineEngine(DecisionEngineEngine):
    def build_feature_store(self, de):
        # Implement logic to create feature groups for search engine based on config
        pass

    def build_models(self, de):
        # Implement logic to create search engine models based on config
        pass

    def build_deployments(self, de):
        # Implement logic to deploy search engine models
        pass
