from abc import ABC, abstractmethod
import humps
import pandas

from hsfs import connection as hsfs_conn
from hsml import connection as hsml_conn


class DecisionEngine(ABC):
    def __init__(self, configs_dict):
        self._name = configs_dict['name']
        self._configs_dict = configs_dict

        self._fs = hsfs_conn().get_feature_store(configs_dict['feature_store'])
        self._mr = hsml_conn().get_model_registry()

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
    def build_deployments(self):
        pass


class RecommendationDecisionEngine(DecisionEngine):
    def build_catalog(self):
        catalog_config = self._configs_dict['catalog_config']

        fg = self._fs.create_feature_group(
            name=catalog_config['feature_group_name'],
            description='Catalog for the Decision Engine project',
            version=1,
            primary_key=catalog_config['primary_key'],
            online_enabled=True,
        )

        df = pandas.read_csv(catalog_config['file_path'])
        fg.insert(df)

    def build_models(self):
        # Implement logic to create recommendation engine models based on config
        pass

    def build_deployments(self):
        # Implement logic to deploy recommendation engine models
        pass


class SearchDecisionEngine(DecisionEngine):
    def __init__(self, config):
        self.config = config

    def build_catalog(self):
        # Implement logic to create feature groups for search engine based on config
        pass

    def build_models(self):
        # Implement logic to create search engine models based on config
        pass

    def build_deployments(self):
        # Implement logic to deploy search engine models
        pass