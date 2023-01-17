from pytorch_lightning.loggers import TensorBoardLogger, CometLogger


def get_loggers(config):
    """
    Function to get training loggers

    Parameters
    ----------
    config: DotMap instance with the current configuration.

    Returns
    -------
    List of loggers
    """
    logger_tb = TensorBoardLogger(config.callbacks.tensorboard_log_dir, name="")
    loggers = [logger_tb]
    if config.comet.enable:
        logger_comet = CometLogger(
            api_key=config.comet.api_key,
            workspace=config.comet.workspace,  # Optional
            project_name=config.comet.project,  # Optional
            experiment_key=config.comet.experiment_key,  # Optional
            experiment_name=config.exp.name,  # Optional
        )
        loggers.append(logger_comet)
    return loggers
