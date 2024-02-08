import yaml


class ConfigReader:
    def __init__(self, config_file: str):
        """Initialize the ConfigReader with the path to the configuration YAML file."""
        self.config_file = config_file

    def read_config(self) -> dict:
        """Read the configuration from the YAML file and return it as a dictionary."""
        with open(self.config_file, "r") as f:
            config = yaml.safe_load(f)
        return config
