# Global Wildfire Prediction using Convolutional Deep Neural Networks

## Project Overview
This repository contains two machine learning projects aimed at processing and analyzing wildfire images to forecast and understand wildfire dynamics. The projects leverage climate data, image processing, and deep neural networks, implemented in PyTorch.

### Key Features
- **Data Transformation**: Conversion of climate data (NC4 format) into 3D matrices incorporating temporal and geospatial information.
- **Image Processing**: Transformation of matrix data into RGB image arrays, highlighting burnt regions.
- **Feature Extraction**: Identification and extraction of 13 key features related to wildfire prediction.
- **Convolution Techniques**: Use of convolution and pooling to enhance feature representation.
- **Neural Network Modeling**: Development of DNN and CNN models with advanced techniques to improve accuracy.

## Directory Structure
The repository is organized into two main folders:

1. **Modeling CNN 10yrs**
   - Contains data processing, neural network coding, model evaluation, and trained models specifically for Convolutional Neural Networks (CNN).
2. **Modeling DNN 10 yrs**
   - Includes similar contents as above but for Deep Neural Networks (DNN).

Each folder contains detailed codes and models developed using PyTorch.

## Implementation Details

### Wildfire Image Processing
- **Technique**: Utilized Numpy for data transformations to create RGB 3-channel image arrays.
- **Objective**: Enhanced differentiation of burnt regions to aid the neural network in image recognition.

### Feature Extraction and Image Convolution
- **Features**: Includes precipitation, burnt area, tree coverage, population density, and lightning frequency.
- **Methods**: Applied convolution, average pooling, data imputation, normalization, and tensor transformations.

### Neural Networks Modeling
- **Frameworks**: Built using PyTorch.
- **Models**: Includes both DNN and CNN with Tanh activation and MSE loss functions.
- **Performance**: Achieved a 75% accuracy on the validation set with CNN, a 5% improvement over standard neural networks.
- **Future Plans**: Exploration of Transformer and RNN architectures for enhanced performance.

### Model Interpretability and Attribution
- **Analysis**: Focused on input features' contribution to wildfire predictions.
- **Key Findings**: Wind force, fuel load, and previous year's burnt area are significant predictors.
