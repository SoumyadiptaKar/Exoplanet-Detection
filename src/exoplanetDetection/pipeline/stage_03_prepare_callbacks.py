from exoplanetDetection.config.configuration import ConfigurationManager
from exoplanetDetection.components.prepare_callbacks import PrepareCallback
from exoplanetDetection.utils.common import logger

STAGE_NAME = "Prepare Base Model stage"

class PrepareCallbacksTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_prepare_callback_config()
        prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = PrepareCallbacksTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<")

    except Exception as e:
        logger.exception(e)
        raise e