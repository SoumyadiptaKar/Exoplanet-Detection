from exoplanetDetection.config.configuration import ConfigurationManager
from exoplanetDetection.components.prepare_callbacks import PrepareCallback
from exoplanetDetection.components.training import Training
from exoplanetDetection.utils.common import logger

STAGE_NAME = "Training Model stage"

class TrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_prepare_callback_config()
        prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        history=training.train(
            callback_list=callback_list
        )
        training.save_loss_curve(history=history)
        training.save_accuracy_curve(history=history)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<")

    except Exception as e:
        logger.exception(e)
        raise e