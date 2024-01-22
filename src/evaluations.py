import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data as Data
from tqdm import tqdm
from sklearn import metrics
import torch
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
from utils import Dataset
from models import network




class Evaluations:

    # === Init function
    def __init__(self,
                 region_dict_dir: str = None,
                 batch_size: int = 20000,
                 split_ratio: float = 0.8,
                 model_param_dir: str = None,
                 parquet_df_dir: str = None,
                 activation: str = None,
                 dropout_value: float = 0.15
                 ) -> None:

        self.region_dict_dir = region_dict_dir
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.model_param_dir = model_param_dir
        self.parquet_df_dir = parquet_df_dir
        self.activation = activation
        self.dropout_value = dropout_value


    def _preprocessing(self):
        # import dataset
        dataset = pd.read_parquet(self.parquet_df_dir)
        # get x and y
        x_indic = dataset[["lat","lon","month","sequence"]]

        x = dataset[['fuel_load_cwdc', 'fuel_load_deadcrootc', 'fuel_wetness', 'fuel_temperature', 'climate_wind', 'climate_tbot', 'climate_rh2m', 'climate_rain', 'human_density', 'light_frequency',"burned_area_mom","burned_area_yoy","burned_area_mom_conv","burned_area_yoy_conv","lat","month"]]
        y = dataset["burned_area"]
        # convert to numpy
        x_indic = np.array(x_indic)
        x = np.array(x)
        y = np.array(y).reshape(-1, 1)

        # standardization
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        x_stand = scaler_X.fit_transform(x)
        y_stand = scaler_y.fit_transform(y)

        # train/test split
        train_x = x_stand[0:int(self.split_ratio*len(x_stand)),:]
        train_y = y_stand[0:int(self.split_ratio*len(y_stand)),:]
        test_x = x_stand[int(self.split_ratio*len(x_stand)):x_stand.shape[0],:]
        test_y = y_stand[int(self.split_ratio*len(y_stand)):y_stand.shape[0],:]

        # Dataloader
        trainset = Dataset(train_x, train_y)
        testset = Dataset(test_x, test_y)
        TrainDataLoader = Data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        TestDataLoader = Data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, drop_last=False)

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
            "train_x": train_x,
        }

        self.net = network(**network_param).to(self.device)

        return x_stand,y_stand,x_indic



    def _load_trained_model(self):
        state_dict = torch.load(self.model_param_dir)
        self.net.load_state_dict(state_dict)

    def _generate_predicts(self, x_stand, y_stand, x_indic):
        # prepare data
        val_x = torch.tensor(x_stand).float()
        val_y = torch.tensor(y_stand).float()
        lat_indicator = x_indic[:,0]
        lon_indicator = x_indic[:,1]
        month_indicator = x_indic[:,2]
        year_indicator = x_indic[:,3]


        # create lists
        self.month_list = []
        self.y_true_list = []
        self.y_pred_list = []
        self.region_list = []
        self.year_list = []

        # validation mode
        self.net.eval()
        for idx in tqdm(range(val_x.shape[0])):
            self.month_list.append(month_indicator[idx])
            self.region_list.append([lat_indicator[idx],lon_indicator[idx]])
            self.year_list.append(year_indicator[idx])
            x_input = val_x[idx,:].to(self.device)
            y_input = val_y[idx,-1].to(self.device).detach().cpu().numpy().squeeze()
            self.y_true_list.append(y_input)
            y_pred = self.net(x_input)
            y_pred = y_pred.detach().cpu().numpy().squeeze()
            self.y_pred_list.append(y_pred)


    def _perform_evaluation(self):
        # Import JSON file
        with open(self.region_dict_dir,'r', encoding='UTF-8') as f:
            region_dict = json.load(f)

        # Lambda Function to substitute Lat and Lon with Region names
        def region_labeler(x):
            for region_name in region_dict.keys():
                if x in region_dict[region_name]:
                    return region_name

        # month_dict_convertor
        month_dict = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        # Lambda Function to substitute years seq to years
        year_dict_temp = list(set(self.year_list))
        year_dict= {}
        sep_lst = []
        start_year = 2001
        for i,v in enumerate(year_dict_temp):
            if (i+1) % 12 != 0:
                sep_lst.append(v)
            else:
                sep_lst.append(v)
                year_dict[start_year] = sep_lst
                sep_lst = []
                start_year += 1
        def year_labeler(x):
            for yrs in year_dict.keys():
                if x in year_dict[yrs]:
                    return yrs

