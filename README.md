# Video Forecasting model (Conv-LSTM) to accelerate the prediction of material microstructure Evolution

<p float="left" align="center"> 
<img src="ms_evolution_example.png" width="90%" />
</p>

## Intro
The focus of this project is the development of cost-effective and adaptable software strategies to promote Virtual Testing for the Process-Structure-Property (PSP) modeling. 

We investigate the use of deep neural networks for the prediction of microstructure evolution in engineering materials. This investigation enables incremental materials technology improvements, by accelerating the traditional CFD workloads and establishing process-structure linkage.

The project is conducted in collaboration with the Materials and Processes Section TEC-MSP at the ESA's Mechanical Engineering Department at ESTEC, Noordwijk, The Netherlands, within the research group of Integrated Computational Materials Engineering for space relevant Additive Manufacturing processes.

## Dataset 
Our dataset consists of 2D time series datasets (image time series). Labels are automatically generated according to statistical sampling (Dakota). Data quality defined by resolution of numerical datasets (correlated to computation time). The time series images are organized in folders, based on different simulation settings and every folder is listed in a `.txt` config file. 
The data are produced and owned by ESA TEC-MSP.

### Dataloader
A data sample (a video) is created based on the temporal sequence of images by a custom dataloader. To enumerate all video samples in the dataset, a new `.txt` config file is manually created that contains a row for each video clip sample in the dataset. Note that the training, validation, and testing datasets have separate config files. Each row is in the format `VIDEO_PATH START_FRAME END_FRAME`. 

A complete tutorial on how to create a video dataloader with pytorch can be found here: [[video dataset loading]](https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch)

## Model 
The model is a conv-lstm network, which is implemented in PyTorch by NVidia Labs, based on the paper, '***Convolutional Tensor-Train LSTM for Spatio-Temporal Learning***', NeurIPS 2020. [[project page](https://sites.google.com/nvidia.com/conv-tt-lstm)] [[original repo](https://github.com/NVlabs/conv-tt-lstm)]


#### License 
Copyright (c) 2020 NVIDIA Corporation. All rights reserved. This work is licensed under a NVIDIA Open Source Non-commercial license.

The code for the conv-tt-lstm model is written by [Wonmin Byeon](https://github.com/wonmin-byeon) \(wbyeon@nvidia.com\) and [Jiahao Su](https://github.com/jiahaosu) \(jiahaosu@terpmail.umd.edu\).


## Contacts
The project for the microstructure prediction is written by [Monica Rotulo](https://github.com/Mooon) \(monica.rotulo@surf.nl\)
