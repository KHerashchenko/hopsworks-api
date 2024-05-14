from abc import ABC, abstractmethod

import os
import numpy as np
import pandas as pd

from hsfs.feature import Feature

from hopsworks.engine import decision_engine_model
import tensorflow as tf

# tf.keras.backend.set_floatx('float64') # didnt solve the error

from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from hsml.transformer import Transformer
from hsfs import embedding


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
        catalog_config = de._configs_dict["product_list"]
        
        ### Creating Items FG ###
        item_features = [
            Feature(name=feat, type=val["type"])
            for feat, val in catalog_config["schema"].items()
        ]
        item_features.append(Feature(name='embeddings', type="ARRAY <double>"))

        emb = embedding.EmbeddingIndex()
        emb.add_embedding(
            "embeddings",            
            de._configs_dict['model_configuration']['retrieval_model']['item_space_dim'], 
        )

        de._items_fg = de._fs.create_feature_group(
            name=de._prefix + catalog_config["feature_view_name"],
            description="Catalog for the Decision Engine project",
            embedding_index=emb,  
            primary_key=[catalog_config["primary_key"]],
            online_enabled=True,
            version=1,
            features=item_features,
        )
        de._items_fg.save()
        
        ### Creating Items FV ###
        items_fv = de._fs.create_feature_view(
            name=de._prefix + catalog_config["feature_view_name"],
            query=de._items_fg.select_all(),
            version=1,
        )

        ### Creating Events FG ###
        events_fg = de._fs.create_feature_group(
            name=de._prefix + "events",
            description="Events stream for the Decision Engine project",
            primary_key=["event_id"],
            online_enabled=True,
            stream=True,
            version=1,
        )

        # Enforced schema for context features in Events FG
        events_features = [
            Feature(name="event_id", type="bigint"),
            Feature(name="session_id", type="string"),
            Feature(name="event_timestamp", type="timestamp"),
            Feature(name="item_id", type="string"),
            Feature(name="event_type", type="string"),
            Feature(
                name="event_value", type="double"
            ),  # e.g. 0 or 1 for click, float for purchase (price)
            Feature(name="event_weight", type="double"),  # event_value multiplier
            Feature(
                name="longitude", type="double"
            ), 
            Feature(name="latitude", type="double"),
            Feature(name="language", type="string"),
            Feature(name="useragent", type="string"),
        ]

        events_fg.save(features=events_features)

        ### Creating Events FV ###
        events_fv = de._fs.create_feature_view(
            name=de._prefix + "events",
            query=events_fg.select_all(),
            version=1,
        )

        td_version, _ = events_fv.create_train_test_split(
            test_size=0.2,
            description="Models training dataset",
            write_options={"wait_for_job": True},
        )

        ### Creating Decisions FG ###
        decisions_fg = de._fs.create_feature_group(
            name=de._prefix + "decisions",
            description="Decisions logging for the Decision Engine project",
            primary_key=["decision_id", "session_id"],
            online_enabled=True,
            version=1,
        )

        # Enforced schema for decisions logging in Decisions FG
        decisions_features = [
            Feature(name="decision_id", type="bigint"),
            Feature(name="session_id", type="string"),
            Feature(
                name="session_activity",
                type="ARRAY <string>",
            ),  # item ids that user interacted with (all event types)
            Feature(
                name="predicted_items",
                type="ARRAY <string>",
            ),  # item ids received by getDecision
        ]
        decisions_fg.save(features=decisions_features)

    def build_models(self, de):
        
        catalog_config = de._configs_dict["product_list"]
        retrieval_config = de._configs_dict["model_configuration"]["retrieval_model"]
        
        # Reading items data into Pandas df
        downloaded_file_path = de._dataset_api.download(
            catalog_config["file_path"], overwrite=True
        )

        de._catalog_df = pd.read_csv(
            downloaded_file_path,
            parse_dates=[
                feat
                for feat, val in catalog_config["schema"].items()
                if val["type"] == "timestamp"
            ],
        )
        de._catalog_df[catalog_config["primary_key"]] = de._catalog_df[
            catalog_config["primary_key"]
        ].astype(str)

        # TODO tensorflow errors if col is of type float64, expecting float32
        # TODO where timestamp feature transformation should happen? (converting into unix format) - it is used in candidate model
        for feat, val in catalog_config["schema"].items():
            if val["type"] == "float":
                de._catalog_df[feat] = de._catalog_df[feat].astype("float32")
            if "transformation" in val.keys() and val["transformation"] == "timestamp":
                de._catalog_df[feat] = de._catalog_df[feat].astype(np.int64) // 10**9
        
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
                
        ### Creating Candidate Model ###
        de._candidate_model = decision_engine_model.ItemCatalogEmbedding(
            de._configs_dict, pk_index_list, categories_lists
        )

        # Adapting features to the items data
        for feat, val in catalog_config["schema"].items():
            if "transformation" not in val.keys():
                continue
            if val["transformation"] in ["numeric", "timestamp"]:
                de._candidate_model.normalized_feats[feat].adapt(
                    de._catalog_df[feat].tolist()
                )
            # elif val["transformation"] == "text":
            #     de._candidate_model.texts_embeddings[feat].layers[0].adapt(
            #         catalog_df[feat].tolist()
            #     )

        tf.saved_model.save(de._candidate_model, "candidate_model")

        # Registering Candidate Model
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
            description="Model that generates embeddings from item features",
            input_example=candidate_example,
            model_schema=candidate_model_schema,
        )
        candidate_model.save("candidate_model")

        ### Creating Query Model ###
        query_model = decision_engine_model.QueryModel(
            vocabulary=pk_index_list, item_space_dim=retrieval_config["item_space_dim"]
        )
        
        # Define the input specifications for the instances
        instances_spec = {
            "context_item_ids": tf.TensorSpec(
                shape=(None, None), dtype=tf.string, name="context_item_ids"
            ),
        }

        # Get the concrete function for the query_model's compute_emb function using the specified input signatures
        signatures = query_model.compute_emb.get_concrete_function(
            instances_spec
        )
        # Save the query_model along with the concrete function signatures
        tf.saved_model.save(
            query_model,
            "query_model",
            signatures=signatures,
        )

        # Registering Query Model
        query_model_schema = ModelSchema(
            input_schema=Schema(
                de._catalog_df.head()[de._configs_dict["product_list"]["primary_key"]]
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

        ### Creating Redirect model for events redirect to Kafka ###
        de._redirect_model = de._mr.python.create_model(
            de._prefix + "events_redirect",
            description="Model for redirecting events into Kafka",
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
        
        items_ds = tf.data.Dataset.from_tensor_slices(
            {col: de._catalog_df[col] for col in de._catalog_df}
        )
        item_embeddings = items_ds.batch(2048).map(
            lambda x: (x[de._configs_dict["product_list"]["primary_key"]], de._candidate_model(x))
        )
        all_embeddings_list = tf.concat([batch[1] for batch in item_embeddings], axis=0).numpy().tolist()
        
        # Reading items data into Pandas df to insert into Items FG
        downloaded_file_path = de._dataset_api.download(
            de._configs_dict["product_list"]["file_path"], overwrite=True
        )
        catalog_df = pd.read_csv(
            downloaded_file_path,
            parse_dates=[
                feat
                for feat, val in de._configs_dict["product_list"]["schema"].items()
                if val["type"] == "timestamp"
            ],
        )
        catalog_df['embeddings'] = all_embeddings_list
        catalog_df[de._configs_dict["product_list"]["primary_key"]] = catalog_df[
            de._configs_dict["product_list"]["primary_key"]
        ].astype(str)
        de._items_fg.insert(catalog_df[list(de._configs_dict["product_list"]["schema"].keys()) + ['embeddings']])

    def build_deployments(self, de):
        # Creating deployment for query model
        mr_query_model = de._mr.get_model(name=de._prefix + "query_model", version=1)

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
            description="Deployment that computes query embedding from session activity and finds closest candidates",
            resources={"num_instances": 1},
            transformer=query_transformer,
        )
        
        # Creating deployment for redirect events endpoint
        redirector_script_path = os.path.join(
            de._redirect_model.version_path, "events_redirect_predictor.py"
        )
        redirect_deployment = de._redirect_model.deploy(
            name=(de._prefix + "events_redirect_deployment").replace("_", "").lower(),
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
        
        retrain_config = de._configs_dict['model_configuration']['retrain']
        cron_schedule = retrain_config['parameter'] if retrain_config['type'] == 'time_based' else "0 0 * * * ?" 
        job.schedule(cron_expression = cron_schedule)

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
        job = de._jobs_api.create_job(de._prefix + "events_consume_job", spark_config)
        job.schedule(cron_expression = "0 0 23 * * ?")  # TODO should run continuously instead 


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
