import json

from hopsworks import decision_engine, client
from hopsworks.client.exceptions import DecisionEngineException


class DecisionEngineApi:
    def __init__(
        self,
        project_id,
        project_name,
    ):
        self._project_id = project_id
        self._project_name = project_name
        
    def create_decision_engine(self, name, config_path):

        dct = {'name': name, 'config_path': config_path}
        _client = client.get_instance()
        path_params = ["project", self._project_id, "python", "decision_engines"]

        return decision_engine.DecisionEngine.from_response_json(
            _client._send_request("POST", path_params, json.dumps(dct)),
            self._project_id,
            self._project_name,
        )

    def _get_decision_engines(self):
        """
        Get all available python decision engines in the project
        """
        _client = client.get_instance()

        path_params = ["project", self._project_id, "python", "decision_engines"]

        return decision_engine.DecisionEngine.from_response_json(
            _client._send_request("GET", path_params),
            self._project_id,
            self._project_name,
        )

    def get_decision_engine(self, name: str):
        """Get decision engine by name.

        # Arguments
            name: name of the decision engine
        # Returns
            `KafkaTopic`: The DecisionEngine object
        # Raises
            `RestAPIError`: If unable to get the topic
        """
        des = self._get_decision_engines()

        for de in des:
            if de.name == name:
                return de

        raise DecisionEngineException("No decision engine named {} could be found".format(name))
