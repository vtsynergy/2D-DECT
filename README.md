# DenseNet and Deconvolution Neural Network (DDnet) for Image enhancement

DDnet is a convolutional neural network that is used for computed tomography image enhancement. The network uses DenseNet blocks for building feature maps and Deconvolution for image reconstruction. DDnet shows superior performance compared to state-of-the-art CT image reconstruction/enhancement algorithms.

## Platform
We developed the framework on the platform shown below:

LSB Version:    :core-4.1-noarch:core-4.1-ppc64le

Distributor ID: CentOS

Description:    CentOS Linux release 7.6.1810 (AltArch) 

Release:        7.6.1810

Codename:       AltArch


## Hardware requirements
The code can run without GPU. Running code with GPU could increase training and inference speed. PyTorch requires Nvidia GPUs with compute capability 6.0 or higher, i.e. any GPU from Pascal, Volta, Turing, Ampere series will work. Our code was tested on Nvidia V100, P100, T4 GPUs.

## Software requirements
The Enhancement AI has been tested on Conda (version: conda 4.6.11), Python (version: 3.6.8), PyTorch (version: 1.0.1), Scikit-image (version: 0.13.1), PIL (version: 5.3.0), Matplotlib (version: 3.0.3), Nibabel (version: 3.2.1), and Cuda compilation tools (release 10.1, V10.1.105)

## Installation
Install Python with anaconda [here](https://docs.anaconda.com/anaconda/install/).

After installing Python, use the package manager [conda](https://docs.conda.io/en/latest/) or [pip](https://pip.pypa.io/en/stable/) to install required packages. 

Conda:
```bash
conda install numpy
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install scikit-image
conda install -c conda-forge pillow
conda install -c conda-forge matplotlib
conda install -c conda-forge nibabel
```

Pip:
```bash
pip install numpy
pip3 install torch torchvision torchaudio
pip install -U scikit-image
pip install --upgrade Pillow
pip install matplotlib
pip install nibabel
``` 
The Cuda compilation tools could be download [here](https://developer.nvidia.com/cuda-downloads). Install the Cuda compilation tool for your platform.

## How to run

1. Following the [Pre-processing Instruction](https://github.com/vtsynergy/2D-DECT/blob/a739ec299051f5b0526202a456994890cdd8e494/Pre-processing_Instruction.md), convert all CT scans to TIFF format (TIFF images must be represented in Hounds Field (HF) unit), and put all CT scans in ../2D-DECT/Images/original_data/. Each scan should be in separate folders.

The folder structure should like shown in below:
```bash
/2D-DECT
  /Images    
    /original_data     
      /scan1
        image1.tif
        image2.tif
        ...
      /scan2
        image1.tif
        image2.tif
        ...
      ...
``` 
2. run ../2D-DECT/code/Integration.py.

```
python  ../2D-DECT/code/Integration.py
```

## Output
Following folders are produced as output from enhancement AI.
1. Images/HU_enhanced: This folder contains enhanced images generated as output from AI. Each scan is put in seperate folders. Each folder contains TIFF images.
2. Images/HU_original: This folder contains the original CT scans in separated folders. Each folder contains TIFF images.
3. Images/HU_enhanced_nii: The folder contains enhanced CT scans in (.nii) format. Each (.nii) file is a 3D lung CT scan.
4. Images/HU_original_nii: The folder contains the original CT scan in (.nii) format. Each (.nii) file is a 3D lung CT scan.
