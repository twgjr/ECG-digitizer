# ECG Image Digitizer
This project began as a data science course project based on the PhyisioNet 2024 Challenge.  The goal is to train a model capable of extracting the signal data from images of electrocardiograms (ECG).  This is a challenging problem worth continuing development.  It was noted that more advanced models and data processing were needed to get good performance on the digitization which is more broadly a task on regression.

This repository contains the code for all the data processing and machine learning model development.

# Preliminary Progress
The initial development was centered around academic understandanding of machine learning in the context of a practical data science project.  As such, the original project team focused on simpler machine learning models.  The alternative model was a typical CNN which processed the image in one shot.  The final report from the prior work is included in this repository for reference. The CNN underfit the data and was not able to reconstruct the ECG waveforms.  One shortcoming of the project was not doing some initial pretraining on simpler ECG images.  Another was to work on the 100Hz downsampled dataset to reduce resource requirements for generating images and training models. 
 Another was using more advanced models to better capture the data.  This project is being continued to address the shortcomings.

# Challenge Results
The Challenge has since ended.  Many participants, including the winner, have published their methods.  The winning team used a UNet with some image preprocessing.  A list of all the papers citing the challenge via Google Scholar:

# Approach
It is believed that using sliding window CNN along with a sequence model may improve performance.  The baseline will be a sliding CNN directly to predict the full singal.  One alternative model will be to use the window CNN as input to an LSTM, then another LSTM to autoregressively decode the signal.  The final alternative will be to apply self attention with a transformer encoder to encode the CNN window, then use a transformer decoder to autoregressively decode the signal.  Each model use the same configurations for the windowed CNN and latent dimesions.

# Data Sets
The dataset being used is the [PTBXL ECG dataset](https://physionet.org/content/ptb-xl/1.0.3/).  It contains 21799 training examples of the raw ecg singal data.  At the time of this writing there there are no available ECG images.  The authors of the challenge collected a real image database called ECG-Image-Databse.  It was not publically available.  Synthetic images were required to be generated with the PhyisioNet ECG Image Kit.  

# Generating ECG Images
[Paper on Generating ECG Images](https://arxiv.org/pdf/2409.16612)

[Python ECG Image Kit](https://github.com/alphanumericslab/ecg-image-kit)

Synthetic images have to be generated with the PhyisioNet ECG Image Kit.  

# Setup
1.  Use a python environment with version 3.12 `conda create -n ecg python=3.12`
2.  Install pip dependencies using the requirements.txt file `pip install -r requirements.txt`
3.  Register the conda environment with jupyter notebook kernel if needed `python -m ipykernel install --user --name=ecg --display-name "ecg"`
