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
        Get all decision engines in the project.
        """
        _client = client.get_instance()

        path_params = ["project", self._project_id, "decisionengine"]

        return decision_engine.DecisionEngine.from_response_json(
            _client._send_request("GET", path_params),
        )

    def get_by_name(self, name):
        """
        Get decision engine by name in the project.
        """
        _client = client.get_instance()

        path_params = ["project", self._project_id, "decisionengine", 'name', name]

        return decision_engine.DecisionEngine.from_response_json(
            _client._send_request("GET", path_params),
        )

    def get_by_id(self, id):
        """
        Get decision engine by id in the project.
        """
        _client = client.get_instance()

        path_params = ["project", self._project_id, "decisionengine", id]

        return decision_engine.DecisionEngine.from_response_json(
            _client._send_request("GET", path_params),
        )

        
    def create(self, de):
        """
        Create decision engine from DecisionEngine object.
        """
        dct = de.json()
        _client = client.get_instance()
        path_params = ["project", self._project_id, "decisionengine"]

        return decision_engine.DecisionEngine.from_response_json(
            _client._send_request("POST", path_params, data=dct)
        )        
        
    def update(self, de):
        """
        Update decision engine from DecisionEngine object.
        """
        dct = de.json()
        _client = client.get_instance()

        path_params = ["project", self._project_id, "decisionengine", de._id]

        return decision_engine.DecisionEngine.from_response_json(
            _client._send_request("PUT", path_params, data=dct),
        )
    
    def delete(self, de):
        """
        Delete decision engine by id.
        """
        _client = client.get_instance()
        path_params = ["project", self._project_id, "decisionengine", de._id]
        _client._send_request("DELETE", path_params)