import mlflow
import functools
import logging

from utils.config import config

logger = logging.getLogger(__name__)

def mlflow_tracking(func):
    """
    Log a LGBM model with MLFlow.
    :param tracking_uri
    :param experiment_name
    :param tags
    :param model_name
    """
    @functools.wraps(func)
    def wrapper_mlflow(*args, **kwargs):
        logger.info("Starting MLFlow tracking..")

        # Set mlflow context
        tracking_uri = config.get_value("tracking_uri")
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"Tracking uri: {tracking_uri}")

        experiment_name = config.get_value("experiment_name")
        if experiment_name is not None:
            experiment = mlflow.create_experiment(experiment_name)
            logger.info(f"Experiment name: {experiment_name}")
        else:
            experiment = mlflow.set_experiment("default experiment")
            logger.info(f"Experiment name set as default")

        tags = config.get_value("tags")
        if tags is not None or len(tags) > 0:
            mlflow.set_tags()

        # Start tracking
        mlflow.start_run(experiment_id=experiment)

        # Run the function
        result = func(*args, *kwargs)

        model = result.get("model")
        metrics = result.get("metrics")

        # Log parameters
        for parameter, value in model.get_params().items():
            try:
                value = float(value or 0)
                mlflow.log_param(parameter, value)
            except (ValueError, TypeError):
                continue

        # Log metrics
        if metrics or len(metrics)>0:
            for metric in metrics:
                mlflow.log_metric(metric, metrics[metric])

        # Log model
        if model:
            model_name = config.get_value("registered_model_name")
            if model_name is not None:
                mlflow.lightgbm.log_model(model, registered_model_name=model_name)
            else:
                mlflow.lightgbm.log_model(model)

        # End tracking
        mlflow.end_run()

        return model
    return wrapper_mlflow
