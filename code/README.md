# DenseNet and Deconvolution Neural Network (DD-Net) for Image enhancement

DD-Net is a convolutional neural network that is used for computed tomography image enhancement. The network uses DenseNet blocks for building feature maps and Deconvolution for image reconstruction. DD-Net shows superior performance compared to state of the art CT image reconstrucion/enhancement algorithms. 

## Installation

Use the package manager [conda](https://docs.conda.io/en/latest/) or [pip](https://pip.pypa.io/en/stable/) to install required packages.

```bash
conda install numpy
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install scikit-image
conda install -c anaconda pillow
```
## Hardware requirements

PyTorch CUDA requires Nvidia GPU with compute capability 6.0 or higher. The network can also run on CPU at low speed.

## How to run

1. Convert all CT scans to TIFF format.
2. Put all CT scans in TIFF format into /enhancement_model/Images/original_data/
3. run /enhancement_model/code/Intergration.py

```
python Intergration.py
```

## Output

1. Images/HU_enhanced: post-enhancement CT scan (.tif) in seperated folders
2. Images/HU_original: pre-enhancement CT scan (.tif) in seperated folders
3. Images/HU_enhanced_nii: post-enhancement 3D CT scan (.nii)
4. Images/HU_original_nii: pre-enhancement 3D CT scan (.nii)
5. Images/Tiff_in_stacks: intermediate result. CT scans containing > 200 images
