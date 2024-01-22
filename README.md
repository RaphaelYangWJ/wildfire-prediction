# Global Wildfire Prediction using Convolutional Deep Neural Networks

## Project Overview
This repository contains two machine learning projects aimed at processing and analyzing wildfire images to forecast and understand wildfire dynamics. The projects leverage climate data, image processing, and deep neural networks, implemented in PyTorch.

## Key Features
- **Data Transformation**: Conversion of climate data (NC4 format) into 3D matrices incorporating temporal and geospatial information.
- **Image Processing**: Transformation of matrix data into RGB image arrays, highlighting burnt regions.
- **Feature Extraction**: Identification and extraction of 13 key features related to wildfire prediction.
- **Convolution Techniques**: Use of convolution and pooling to enhance feature representation.
- **Neural Network Modeling**: Development of DNN and CNN models with advanced techniques to improve accuracy.
- Model Attribution Analysis: Parse the network at neuron level to analyze the network.

## Prerequisites

List any prerequisites, libraries, OS version, etc., needed before installing your project.

```
numpy == 1.26.1
pandas == 2.1.1
sklearn == 0.0.10
torch == 2.1.0
torchvision == 4.35.0
matplotlib == 3.8.0
tqdm == 4.65.2
omegaconf == 2.3.0
seaborn == 0.13.0
tqdm == 4.65.2
captum == 0.7.0
```



## Directory Structure

The repository is organized into two main folders:

1. `/src`: All the source codes for the training pipeline.
   1. `/src/data_engineering.py`: Process the source data, perform convolutions on data
   2. `/src/models.py`: Model architecture
   3. `/src/trainer.py`: Training process script
   4. `/src/utils.py`: Pytorch Dataset class for Data loader
   5. `/src/main.py`: Training pipeline access

2. `/notebooks`: Documentation files for the project.
   1. `/notebooks/Notebook - Raw Data Process [Obsoleted].py`: Obsoleted
   2. `/notebooks/Notebook - Training Pipeline [Obsoleted].py`: Obsoleted
   3. `/notebooks/Notebook - Data Engineering [Obsoleted].py`: Obsoleted
   4. `/notebooks/Notebook - Evaluation Pipeline.py`: Jupyter Notebook for Evaluation Pipeline
   5. `/notebooks/Notebook - Attribution Analysis.py`: Jupyter Notebook for Attribution Analysis

3. `/config`: Set your configurations and parameters for training pipeline, evaluation and attribution analysis
4. `/dataset`: dataset processed is saved here.
5. `/raw_dataset`: Raw dataset is saved here.
6. `/trained_model`: Training Pipeline running result will be automatically saved here sequentially.

## Implementation Details

### 1. Wildfire Image Processing

- **Technique**: Utilized Numpy for data transformations to create RGB 3-channel image arrays.
- **Objective**: Enhanced differentiation of burnt regions to aid the neural network in image recognition.

### 2. Feature Extraction and Image Convolution
- **Features**: Includes precipitation, burnt area, tree coverage, population density, and lightning frequency.
- **Methods**: Applied convolution, average pooling, data imputation, normalization, and tensor transformations.

### 3. Neural Networks Modeling
- **Frameworks**: Built using PyTorch.
- **Models**: Includes both DNN and CNN with Tanh activation and MSE loss functions.
- **Performance**: Achieved a 75% accuracy on the validation set with CNN, a 5% improvement over standard neural networks.
- **Future Plans**: Exploration of Transformer and RNN architectures for enhanced performance.

### 4. Model Interpretability and Attribution
- **Analysis**: Focused on input features' contribution to wildfire predictions.
- **Key Findings**: Wind force, fuel load, and previous year's burnt area are significant predictors.

## Acknowledgments

- **Project Supervisor**: Berkeley Lawrence National Lab, Qing Zhu
- **Project Researcher**: University of California, Berkeley, Weijie Yang
