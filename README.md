# Video Forecasting model (Conv-TT-LSTM) applied to Microstructure Evolution

## Requirements
- Python > 3.0
- Pytorch 1.0

## Install 
pip3 install -r requirements.txt

## Dataset
Our dataset consists of 2D time series datasets (image time series). Dataset provided and owned by ESA TEC-MSP. In order to feed the training/testing samples to the model, a video samples has to be created.

## Model
PyTorch implementations of the paper, '***Convolutional Tensor-Train LSTM for Spatio-Temporal Learning***', NeurIPS 2020. [[project page](https://sites.google.com/nvidia.com/conv-tt-lstm)]

* Original implementation to reproduce the numbers in the paper.
* We also provide a highly optimized implementation for NVIDIA GPU under the folder ../code_opt/, check it out! 

Original implementation tested on:
- MNIST
- KTH
datasets.

#### License 
Copyright (c) 2020 NVIDIA Corporation. All rights reserved. This work is licensed under a NVIDIA Open Source Non-commercial license.


## Loading the dataset and training the model
1) create a custom config file for training and testing with ``config_videocreator.py``
2) Load the videos with ``ms_dataloader.py``
3) Train the model with ``ms_model_train.py``

## Training the model with HPC cluster
In order to submit an already created job script, the command ``sbatch`` has to be used.
```
sbatch train_ms.sh
```

## Testing the model
* In order to run the test code, you must have trained the model and saved at least one checkpoint as a '.pt'  file


## Contacts
The microstructure project is written by [Monica Rotulo](https://github.com/mooon) \(monica.rotulo@gmail.com\).

The conv-lstm/conv-tt-lstm code was written by [Wonmin Byeon](https://github.com/wonmin-byeon) \(wbyeon@nvidia.com\) and [Jiahao Su](https://github.com/jiahaosu) \(jiahaosu@terpmail.umd.edu\).
