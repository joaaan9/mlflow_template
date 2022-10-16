from pathlib import PureWindowsPath, Path

import mlflow
from functools import wraps
import logging

from utils.config import config

logger = logging.getLogger(__name__)


def mlflow_tracking(func):
    """
    Tracking project with MLFlow.
    """
    @wraps(func)
    def wrapper_mlflow():

        logger.info("Starting MLFlow tracking..")

        # Set mlflow context
        tracking_uri = config.get_var("mlflow")["tracking_uri"]
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"Tracking uri: {tracking_uri}")

        experiment_name = config.get_var("mlflow")["experiment_name"]
        if experiment_name is not None:
            experiment = mlflow.set_experiment(experiment_name=experiment_name)
            logger.info(f"Experiment name: {experiment_name}")
        else:
            experiment = mlflow.set_experiment("default experiment")
            logger.info(f"Experiment name set as default")

        tags = config.get_var("mlflow")["tags"]
        if tags is not None:
            mlflow.set_tags()

        # Start tracking
        mlflow.start_run(experiment_id=experiment.experiment_id)

        # Run the function
        result = func()

        # Log extra params of config
        extra = config.get_var("extra")[0]
        for key in extra.keys():
            mlflow.log_param(key, extra[key])

        # Log parameters
        if "model" in result.keys():
            model = result["model"]
            for parameter, value in model.get_params().items():
                try:
                    value = float(value or 0)
                    mlflow.log_param(parameter, value)
                except (ValueError, TypeError):
                    continue

            # Log model
            model_name = config.get_var("mlflow")["registered_model_name"]
            if model_name is not None:
                mlflow.lightgbm.log_model(model, "model", registered_model_name=model_name)
            else:
                mlflow.lightgbm.log_model(model, "model")

        # Log metrics
        if "metrics" in result.keys():
            for metric in result["metrics"]:
                mlflow.log_metric(metric, result["metrics"][metric])

        if "artifacts" in result.keys():
            mlflow.log_artifact(result["artifacts"], "results")

        # End tracking
        mlflow.end_run()

    return wrapper_mlflow
