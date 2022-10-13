import os
import yaml
import logging
from pathlib import Path, PureWindowsPath


class config:
    @staticmethod
    def load_conf():
        path = PureWindowsPath(Path(__file__)).parents[0]
        path_config = os.path.join(path, "../config.yaml")
        if os.path.exists(path_config):
            with open(path_config) as file:
                config = yaml.safe_load(file)
                if "current_config" in config:
                    model_name = config["current_config"]
                else:
                    logging.warning("Current config is not set!")
                if model_name in config:
                    parameters = config[model_name]
                else:
                    logging.warning("Model on current_config doesn't exist!")
        else:
            logging.warning("File config.yaml doesn't exist!")
        return parameters

    def get_var(key: str):
        parameters = config.load_conf()
        var = parameters.get(key)
        if var is not None:
            return var
        else:
            logging.warning("This config variable doesn't exist!")


if __name__ == '__main__':
    vars = ["name", "sql", "target_metric", "registered_model_name", "extra"]
    # conf = config()
    for var in vars:
        print(config.get_var(var))