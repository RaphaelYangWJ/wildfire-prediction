import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from utils import Dataset
from models import network
from sklearn import metrics
import torch.optim as optim
from torchinfo import summary
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler




# === Class: Trainer
class Trainer:
    # === Init function
    def __init__(self,
                 seed_val: int = 1,
                 data_dir: str = None,
                 batch_size: int = 20000,
                 split_ratio: float = 0.8,
                 learning_rate: float = 0.00015,
                 epochs: int = 300,
                 activation: str = "Tanh",
                 dropout_value: float = 0.1,
                 ) -> None:

        self.seed_val = seed_val
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = activation
        self.dropout_value = dropout_value
        self.msg = "Training Parameters:\n" + \
                   "- Batch Size:\n" + str(batch_size) + "\n" + \
                   "- Train Test Split:\n" + str(split_ratio) + "\n" + \
                   "- Learning Rate:\n" + str(learning_rate) + "\n" + \
                   "- Epochs:\n" + str(epochs) + "\n" + \
                   "- Activation Function:\n" + str(activation) + "\n"

    # === Function: Load dataset
    def _load_dataset(self):
        print("Loading feature dataset...")
        # import dataset
        dataset = pd.read_parquet(self.data_dir + "feature_dataset.parquet")
        print(dataset.columns)
        # get x and y
        x = dataset[['fuel_load_cwdc', 'fuel_load_deadcrootc', 'fuel_wetness', 'fuel_temperature', 'climate_wind',
                     'climate_tbot', 'climate_rh2m', 'climate_rain', 'human_density', 'light_frequency',
                     "burned_area_mom","burned_area_yoy","burned_area_mom_conv","burned_area_yoy_conv","lat","month"]]
        y = dataset["burned_area"]
        # convert to numpy
        x = np.array(x)
        y = np.array(y).reshape(-1, 1)
        print(x.shape, y.shape)
        return x, y

    # === Function: Train/Test split
    def _train_test_split(self, x, y):
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        x_stand = scaler_X.fit_transform(x)
        y_stand = scaler_y.fit_transform(y)
        train_x = x_stand[0:int(self.split_ratio*len(x_stand)),:]
        train_y = y_stand[0:int(self.split_ratio*len(y_stand)),:]
        test_x = x_stand[int(self.split_ratio*len(x_stand)):x_stand.shape[0],:]
        test_y = y_stand[int(self.split_ratio*len(y_stand)):y_stand.shape[0],:]
        self.train_x = train_x
        return train_x, train_y, test_x, test_y

    # === Function: DataLoader Preparation
    def _dataloader(self, train_x, train_y, test_x, test_y):
        trainset = Dataset(train_x, train_y)
        testset = Dataset(test_x, test_y)
        self.TrainDataLoader = Data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.TestDataLoader = Data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, drop_last=False)

    # === Function: Load Network
    def _load_network(self):
        # set seed
        torch.cuda.manual_seed(self.seed_val)
        # Switch to GPU for Training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Clear GPU caches
        torch.cuda.empty_cache()
        # load net
        network_param = {
                            "activation": self.activation,
                            "dropout_value": self.dropout_value,
                            "train_x": self.train_x,
                        }

        self.net = network(**network_param).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.msg = self.msg + str(summary(self.net)) + "\n"

    # === Function: Train the Network
    def _fit_network(self):
        # metrics lists
        train_loss_history = []
        train_r2_history = []
        val_loss_history = []
        val_r2_history = []
        val_rho_history = []
        record_acc = 0


        # Perform iterations
        for epoch in tqdm(range(self.epochs), desc='Training'):
            # create lists
            train_loss_lst = []
            train_r2_lst = []
            val_loss_lst = []
            val_r2_lst = []
            val_rho_lst = []


            # ====== Training Mode ======
            self.net.train()
            for (x_input, y_true) in self.TrainDataLoader:
                # attach to GPU
                x_input = x_input.to(self.device)
                y_true = y_true.to(self.device)
                # gain outputs
                outputs = self.net(x_input)
                # compute loss
                loss = self.criterion(outputs, y_true)
                # Back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # r2 = r_squared(y_true, outputs)
                r2 = metrics.r2_score(y_true.detach().cpu().numpy().squeeze(),outputs.detach().cpu().numpy().squeeze())
                # append results
                train_loss_lst.append(loss.item())
                train_r2_lst.append(r2)
            # Append the performance result after each epoch
            best_perform = train_r2_lst.index(max(train_r2_lst))
            train_loss_history.append(train_loss_lst[best_perform])
            train_r2_history.append(train_r2_lst[best_perform])
            # ====== Training Mode End ======


            # ====== Testing Mode ======
            self.net.eval()
            for (x_val_input, y_val_input) in self.TestDataLoader:
                # attach to GPU
                x_val_input = x_val_input.to(self.device)
                y_val_input = y_val_input.to(self.device)
                # gain outputs
                outputs = self.net(x_val_input)
                # compute loss
                loss = self.criterion(outputs, y_val_input)
                # r2 = r_squared(y_val_input, outputs) & Rho
                r2 = metrics.r2_score(y_val_input.detach().cpu().numpy().squeeze(),outputs.detach().cpu().numpy().squeeze())
                rho = np.corrcoef(y_val_input.detach().cpu().numpy().squeeze(), outputs.detach().cpu().numpy().squeeze())[0][1]
                # append results
                val_rho_lst.append(rho)
                val_loss_lst.append(loss.item())
                val_r2_lst.append(r2)
            # Append the performance result after each epoch
            best_perform = val_r2_lst.index(max(val_r2_lst))
            val_loss_history.append(val_loss_lst[best_perform])
            val_r2_history.append(val_r2_lst[best_perform])
            val_rho_history.append(val_rho_lst[best_perform])
            # ====== Testing Mode End ======

            # save best performance model
            if best_perform >= record_acc:
                record_acc = best_perform
                self._save_checkpoint()

            # output the performance every 10 times
            if str(epoch+1).endswith("0"):
                msg_info = "Current Epoch: "+str(epoch+1)+"/"+str(self.epochs) + \
                      " : Train Loss: "+str(train_loss_history[-1]) + \
                      " | Train R2: "+str(train_r2_history[-1]) + \
                      " | Val Loss: "+str(val_loss_history[-1]) + \
                      " | Val Rho: "+str(val_rho_history[-1]) + \
                      " | Val R2: "+str(val_r2_history[-1])
                print("\n"+msg_info)
                self.msg = self.msg + msg_info + "\n"

        return self.msg


        # save performance
        self._save_performance(train_loss_history, train_r2_history, val_loss_history, val_r2_history, val_rho_history)

    # === Function: Save Model
    def _save_checkpoint(self):
        path = self.data_dir + "trained_model.pth"
        torch.save(self.net.state_dict(),path)

    # === Function: Save Performance
    def _save_performance(self, train_loss_history, train_r2_history, val_loss_history, val_r2_history, val_rho_history):
        evaluation = pd.DataFrame(columns = ["train loss","train accuracy","validation loss","validation accuracy","validation Rho"])
        evaluation["train loss"] = train_loss_history
        evaluation["train accuracy"] = train_r2_history
        evaluation["validation loss"] = val_loss_history
        evaluation["validation accuracy"] = val_r2_history
        evaluation["validation Rho"] = val_rho_history
        evaluation.to_csv(self.data_dir + "model_cnn_evaluation.csv",index=False)

        plt.plot(train_loss_history, label = "train loss")
        plt.plot(val_loss_history, label = "test loss")
        plt.title("Loss Diagram of Model")
        plt.legend()
        plt.savefig(self.data_dir + "loss diagram.png")

        plt.plot(train_r2_history, label = "train accuracy")
        plt.plot(val_r2_history, label = "test accuracy")
        plt.title("Accuracy (R-squared) Diagram of Model")
        plt.legend()
        plt.savefig(self.data_dir + "accuracy diagram.png")

        plt.plot(val_rho_history)
        plt.title("Correlation between prediction & true burned area")
        plt.savefig(self.data_dir + "correlation diagram.png")