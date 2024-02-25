import torch
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from GCN_CBAM import Convolutioner
warnings.filterwarnings("ignore")





# Class: Data Engineering
class Data_Engineering:

    # === Init function
    def __init__(self,
                 data_path: str = None,
                 year_len: int = 10,
                 conv_method: str = "max",
                 save_dir: str = None
                 ) -> None:


        self.data_path = data_path
        self.year_len = year_len
        self.conv_method = conv_method
        self.save_dir = save_dir
        self.gcn_cnn_conv = Convolutioner(features = -1, out_dim = 9, kernel_size = 7)


    # === Function: Load dataset
    def _load_data(self):
        # Import features
        self.fuel_load_cwdc = np.load(file = self.data_path + "fuel_load_cwdc.npy")
        self.fuel_load_deadcrootc = np.load(file = self.data_path + "fuel_load_deadcrootc.npy")
        self.fuel_wetness = np.load(file = self.data_path + "fuel_wetness.npy")
        self.fuel_temperature = np.load(file = self.data_path + "fuel_temperature.npy")
        self.climate_wind = np.load(file = self.data_path + "climate_wind.npy")
        self.climate_tbot = np.load(file = self.data_path + "climate_tbot.npy")
        self.climate_rh2m = np.load(file = self.data_path + "climate_rh2m.npy")
        self.climate_rain = np.load(file = self.data_path + "climate_rain.npy")
        self.tree_coverage = np.load(file = self.data_path + "tree_coverage.npy")
        self.human_density = np.load(file = self.data_path + "human_density.npy")
        self.light_frequency = np.load(file = self.data_path + "light_frequency.npy")
        # import output
        self.burned_area = np.load(file = self.data_path + "burned_area.npy")
        self.percentage_burned_area = np.load(file = self.data_path + "percent_burned_area.npy")
        self.grid_area = np.load(file = self.data_path + "grid_area.npy")

    # === Function: Filter valid grids
    def _valid_grid_filter(self):
        self.valid_grid = []
        for i in range(96):
            for j in range(143):
                if np.sum(self.percentage_burned_area[:,i,j]) > 0:
                    self.valid_grid.append([i,j])

    # === Function: Generate data
    def _data_generation(self):
        # === data time range
        year_indic = 1800 - self.year_len * 12
        # Create a Dataframe
        data = pd.DataFrame()

        # Iterate to get the data
        for i in tqdm(range(len(self.valid_grid)), desc='Processing'):
            index_value = self.valid_grid[i]
            # dataframes
            temp_df = pd.DataFrame()

            # === Basic Features
            temp_df["fuel_load_cwdc"] = self.fuel_load_cwdc[year_indic:,index_value[0],index_value[1]].tolist()
            temp_df["fuel_load_deadcrootc"] = self.fuel_load_deadcrootc[year_indic:,index_value[0],index_value[1]].tolist()
            temp_df["fuel_wetness"] = self.fuel_wetness[year_indic:,index_value[0],index_value[1]].tolist()
            temp_df["fuel_temperature"] = self.fuel_temperature[year_indic:,index_value[0],index_value[1]].tolist()
            temp_df["climate_wind"] = self.climate_wind[year_indic:,index_value[0],index_value[1]].tolist()
            temp_df["climate_tbot"] = self.climate_tbot[year_indic:,index_value[0],index_value[1]].tolist()
            temp_df["climate_rh2m"] = self.climate_rh2m[year_indic:,index_value[0],index_value[1]].tolist()
            temp_df["climate_rain"] = self.climate_rain[year_indic:,index_value[0],index_value[1]].tolist()
            temp_df["human_density"] = self.human_density[year_indic:,index_value[0],index_value[1]].tolist()
            temp_df["light_frequency"] = self.light_frequency[year_indic:,index_value[0],index_value[1]].tolist()
            # temp_df["tree_coverage"] = tree_coverage[:,10,index_value[0],index_value[1]].tolist()

            # === Label
            temp_df["burned_area"] = self.burned_area[year_indic:,index_value[0],index_value[1]].tolist()

            # === Additional Features
            temp_df["burned_area_mom"] = temp_df["burned_area"].shift()
            temp_df["burned_area_yoy"] = temp_df["burned_area"].shift(12)
            temp_df["month"] = [12,1,2,3,4,5,6,7,8,9,10,11] * self.year_len
            temp_df["sequence"] = [i for i in range(1800 - year_indic)]
            temp_df["lat"] = index_value[0]
            temp_df["lon"] = index_value[1]
            # === drop zero values
            # temp_df = temp_df[temp_df["burned_area"]!=0]

            # === Convolutional on Features of "burned_area_mom" & "burned_area_yoy"
            temp_join_list = []
            conv_burned_area = self.burned_area[year_indic:,(index_value[0]-1):(index_value[0]+2),(index_value[1]-1):(index_value[1]+2)]
            # judge the shape
            if conv_burned_area.shape[1] == 3 and conv_burned_area.shape[2] == 3:
                for i in range(120):
                    value_temp = torch.Tensor([conv_burned_area[i]])
                    value_temp = self.gcn_cnn_conv(value_temp)
                    temp_join_list.append(value_temp)
            else:
                for i in range(120):
                    temp_join_list.append(None)


            # === load convolution values to list
            temp_df["burned_area_conv"] = temp_join_list
            temp_df["burned_area_mom_conv"] = temp_df["burned_area_conv"].shift()
            temp_df["burned_area_yoy_conv"] = temp_df["burned_area_conv"].shift(12)
            del temp_df["burned_area_conv"]

            # === only leave useful sequences
            # temp_df = temp_df.loc[temp_df["sequence"] >= year_indic]

            # === trim out zeros for burned areas
            temp_df["burned_area"] = temp_df["burned_area"].apply(lambda x: None if x == 0 else x)
            temp_df = temp_df.dropna(axis = 0, subset=["burned_area","burned_area_mom","burned_area_yoy","burned_area_mom_conv","burned_area_yoy_conv"])

            # === concat to data
            data = pd.concat([data,temp_df])
        # === return data
        return data

    # === Function: Preprocessing
    def _preprocessing(self, df):
        df['burned_area'] = df['burned_area'].apply(lambda x: np.log(x))
        df['burned_area_mom'] = df['burned_area_mom'].apply(lambda x: np.log(x))
        df['burned_area_yoy'] = df['burned_area_yoy'].apply(lambda x: np.log(x))
        df['burned_area_mom_conv'] = df['burned_area_mom_conv'].apply(lambda x: np.log(x))
        df['burned_area_yoy_conv'] = df['burned_area_yoy_conv'].apply(lambda x: np.log(x))
        df.replace([np.inf, -np.inf], None, inplace=True)
        df.dropna(axis = 0, inplace = True)
        df = df.loc[df["burned_area"] > 0]
        # === heatmap
        fig = sns.heatmap(df.corr(), vmax=.8, square=True, annot=True, annot_kws={"size":4}, cmap="Blues")
        fig.figure.savefig(self.save_dir + "features heatmap")
        # === save the prepared dataset
        df.to_parquet(self.save_dir + "feature_dataset.parquet", index = None)

    # === Function: Processing Pipeline
    def processing_pipeline(self):
        print("+++++ Processing Dataset....")
        msg = "Dataset Processing Parameters:\n"+ \
              "- Time Period: " + str(self.year_len) + "\n" + \
              "- Conv Method: " + str(self.conv_method)
        print(msg)
        self._load_data()
        self._valid_grid_filter()
        pro_data = self._data_generation()
        self._preprocessing(pro_data)
        return msg