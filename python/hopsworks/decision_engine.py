import json
import humps

from hopsworks import client
from hopsworks.core import opensearch_api, dataset_api, kafka_api, job_api
from hopsworks.engine import decision_engine_engine
import yaml

from hsfs import connection as hsfs_conn
from hsml import connection as hsml_conn

# tf.keras.backend.set_floatx('float64') # didnt solve the error

class DecisionEngine():
    def __init__(self, name, config_file_path, use_case, job_name=None, id=None, *args, **kwargs):
        self._id = id
        self._name = name
        self._use_case = use_case
        self._config_file_path = config_file_path
        self._job_name = job_name
        
        self._prefix = "de_" + self._name + "_"
        self._catalog_df = None
        self._items_fg = None
        self._retrieval_model = None
        self._redirect_model = None

        client.init("hopsworks")
        self._client = client.get_instance()
        self._opensearch_api = opensearch_api.OpenSearchApi(
            self._client._project_id, self._client._project_name
        )
        self._dataset_api = dataset_api.DatasetApi(self._client._project_id)
        self._configs_dict = self.load_configs(self._config_file_path)
        
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

        if self._configs_dict['use_case'] == "recommendation":
            self._decision_engine_engine = decision_engine_engine.RecommendationDecisionEngineEngine()
        elif self._configs_dict['use_case'] == "search":
            self._decision_engine_engine = decision_engine_engine.SearchDecisionEngineEngine()
            

    @classmethod
    def from_response_json(cls, json_dict):
        json_decamelized = humps.decamelize(json_dict)
        if "count" not in json_decamelized:
            return DecisionEngine(
                **json_decamelized
            )
        elif json_decamelized["count"] == 0:
            return []
        else:
            return [
                DecisionEngine(
                    **decision_engine
                )
                for decision_engine in json_decamelized["items"]
            ]

    def update_from_response_json(self, json_dict):
        json_decamelized = humps.decamelize(json_dict)
        self.__init__(**json_decamelized)
        return self

    def load_configs(self, file_path):
        downloaded_file_path = self._dataset_api.download(file_path, overwrite=True)
        with open(downloaded_file_path, 'r') as yaml_file:
            configs_dict = yaml.safe_load(yaml_file)
        return configs_dict
            
    def to_dict(self):
        return humps.camelize({"name": self._name, "config_file_path": self._config_file_path, "use_case": self._use_case.upper()})

    def json(self) -> str:
        return json.dumps(self.to_dict())
    
    def setup(self):
        self._decision_engine_engine.setup_decision_engine(self)

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

