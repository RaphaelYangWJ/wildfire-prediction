import os
import logging
import omegaconf
from trainer import Trainer
from gcn_engineering import Data_Engineering



# === Class: Main Pipeline
class main_pipeline(object):

    # === Function: Init
    def __init__(self, config):
        self.config = config
        # Create save directory
        existing_models = len(os.listdir(self.config.save_dir))
        self.save_dir = self.config.save_dir + "Train-Result-" + str(existing_models)
        os.makedirs(self.save_dir)
        self.save_dir += "/"

    # === Function: Logging save
    def save_logging(self):
        logging.basicConfig(
            filename = self.save_dir+"training_log.log",
            level=logging.INFO,
            format="%(asctime)s > %(message)s",
        )

    # === Function: Data Engineering Setup
    def data_engineering_setup(self):
        # init dataset
        params = {
                    "data_path": self.config.data_engineer.data_path,
                    "year_len": self.config.data_engineer.year_len,
                    "conv_method": self.config.data_engineer.conv_method,
                    "save_dir": self.save_dir,
                }
        #
        processor = Data_Engineering(**params)
        msg = processor.processing_pipeline()
        logging.info(msg)

    # === Function: Training Setup
    def training_setup(self):
        # model setup
        params = {
            "seed_val": self.config.model.seed,
            "data_dir": self.save_dir,
            "batch_size": self.config.trainer.batch_size,
            "split_ratio": self.config.trainer.split_ratio,
            "learning_rate": self.config.trainer.learning_rate,
            "epochs": self.config.trainer.epochs,
            "activation": self.config.model.activation,
            "dropout_value": self.config.model.dropout_value,
        }

        print("+++++ Training Setup....")

        training = Trainer(**params)
        x, y = training._load_dataset()
        train_x, train_y, test_x, test_y = training._train_test_split(x, y)
        training._dataloader(train_x, train_y, test_x, test_y)
        training._load_network()
        msg = training._fit_network()
        logging.info(msg)

    # === Function: Pipeline Setup
    def pipeline_setup(self):
        self.save_logging()
        self.data_engineering_setup()
        self.training_setup()




# === Training Pipeline Access
if __name__ == "__main__":
    # === Load configuration file
    config_path = "../config/config.yaml"
    config = omegaconf.OmegaConf.load(config_path)

    # === Training initiation
    train_model = main_pipeline(config)
    train_model.pipeline_setup()