from hopsworks import decision_engine, client

class DecisionEngineApi:
    def __init__(
        self,
        project_id,
        project_name,
    ):
        self._project_id = project_id
        self._project_name = project_name
        
    def get_all(self):
        """
        Get all available python decision engines in the project
        """
        _client = client.get_instance()

        path_params = ["project", self._project_id, "decisionengine"]

        return decision_engine.DecisionEngine.from_response_json(
            _client._send_request("GET", path_params),
        )

    def get_by_name(self, name):
        """
        Get decision engine by name in the project
        """
        _client = client.get_instance()

        path_params = ["project", self._project_id, "decisionengine", 'name', name]

        return decision_engine.DecisionEngine.from_response_json(
            _client._send_request("GET", path_params),
        )

    def get_by_id(self, id):
        """
        Get decision engine by id in the project
        """
        _client = client.get_instance()

        path_params = ["project", self._project_id, "decisionengine", id]

        return decision_engine.DecisionEngine.from_response_json(
            _client._send_request("GET", path_params),
        )

        
    def create(self, decision_engine):
        """
        Create from DecisionEngine object
        """
        dct = decision_engine.json()
        _client = client.get_instance()
        path_params = ["project", self._project_id, "decisionengine"]

        return decision_engine.DecisionEngine.from_response_json(
            _client._send_request("POST", path_params, data=dct)
        )        
        
    def update(self, decision_engine):
        """
        Get all available python decision engines in the project
        """
        dct = decision_engine.json()
        _client = client.get_instance()

        path_params = ["project", self._project_id, "decisionengine", decision_engine._id]

        return decision_engine.DecisionEngine.from_response_json(
            _client._send_request("PUT", path_params, data=dct),
        )
    
    def delete(self, decision_engine):
        """
        Delete decision engine by id.
        """
        _client = client.get_instance()
        path_params = ["project", self._project_id, "decisionengine", decision_engine._id]
        _client._send_request("DELETE", path_params)