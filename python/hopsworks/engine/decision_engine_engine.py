from hopsworks.decision_engine import RecommendationDecisionEngine, SearchDecisionEngine


class DecisionEngineEngine:
    def __init__(self):
        pass

    def setup_decision_engine(self, configs_dict):

        # Determine the use case from config and instantiate the appropriate builder
        if configs_dict['use_case'] == 'recommendation_engine':
            builder = RecommendationDecisionEngine(configs_dict)
        elif configs_dict['use_case'] == 'search_engine':
            builder = SearchDecisionEngine(configs_dict)
        else:
            raise ValueError("Invalid use case specified in config")

        # # Build project components using the builder
        builder.build_catalog()
        builder.build_models()
        builder.build_vector_db()
        builder.build_deployments()
