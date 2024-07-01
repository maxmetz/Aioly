Deep Learning for spectroscopic Data:

Application with (OSSL) Open Soil Spectral Library soil NIR/MIR Database: 
https://soilspectroscopy.org/introducing-the-open-soil-spectral-library/

Load_dataset:
A dataset class for OSSL data adapted for torch dataloader.

base_net:
define two networks CuiNet and DeepSpectraCNN

example of training:
train_CuiNet
train_DeepSectra



CuiNet is derived from:
Modern practical convolutional neural networks for multivariate regression: Applications to NIR calibration
Chenhao Cui, Tom Fearn
DOI: 10.1016/j.chemolab.2018.07.008 
2018


DeepSpectraCNN is derived from:
DeepSpectra: An end-to-end deep learning approach for quantitative spectral analysis
Xiaolei Zhang and Tao Lin and Jinfan Xu and Xuan Luo and Yibin Ying
https://doi.org/10.1016/j.aca.2019.01.002
2019
