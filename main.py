from exoplanetDetection.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from exoplanetDetection.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from exoplanetDetection.pipeline.stage_03_training import TrainingPipeline
from exoplanetDetection import logger


STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare Base Model stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Training stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    prepare_callbacks = TrainingPipeline()
    prepare_callbacks.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e