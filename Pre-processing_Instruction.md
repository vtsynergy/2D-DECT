# Pre-processing Instruction

Instruction of where to download the CT image data for this framework and how to do the pre-processing of the original source images. 

## Software Requirement
ImageJ. Go to [here](https://imagej.nih.gov/ij/download.html) to install ImageJ on your computer.

![image](https://user-images.githubusercontent.com/31482058/116592443-0258f800-a8ee-11eb-83ee-3068d6c5672a.png)

## Download CT iamge data of COVID-19 (BIMCV+ data for example)
Go to the open-source database BIMCV-COVID19+ Database at [here](https://osf.io/nh7g8/files/) to download CT image data. (Expand "Data" -->"V1.0" and download subject files)

![image](https://user-images.githubusercontent.com/31482058/116592664-3fbd8580-a8ee-11eb-96c2-12c9e6d02506.png)

For example, download "bimcv_covid19_posi_subjects_17.tgz". Unzip the downloaded tgz file, it will generate 40 folders, from "sub-S03693" to "sub-S03734". Search the files whose name is end with ".nii.gz" in upziped folders. For example, "/sub-S03693/ses-E08061/mod-rx/sub-S03693_ses-E08061_run-2_bp-chest_ct.nii.gz" is a standard 3D lung CT images. (The size of meaningful 3D CT image is generally larger than 1MB.)

Offer a [help function](https://github.com/vtsynergy/2D-DECT/blob/main/code/nii_to_tiff.py) to convert .nii.gz to .tif file.
## Pre-processing Procedure

**1. Pick AXIAL CT image stacks**

1.1. Unzip the downloaded “tgz” file into a new folder. Go to every one of the subfolders within the new folder, and manually inspect each image file (by dragging and dropping the image file onto the ImageJ window).

![image](https://user-images.githubusercontent.com/31482058/116592934-90cd7980-a8ee-11eb-85f8-d6b81429a72c.png)

1.2. If you see an image file has axial CT slice images, copy this image file to another new folder that can be named such as “Axial CT images”. This new folder will accumulate all the original axial CT images from the BIMCV-COVID-19+ database, and should be placed somewhere so you can easily find, for example your desktop. This folder will serve as the “original” reference.

**2. Orientate CT images**

For each correctly identified axial CT image file from Step 2, manually do the following:

2.1. Align all the CT slices in the file so that the slice image’s top corresponds to the patient’s chest, bottom corresponds to the patient’s back, left corresponds to the patient’s left, and right corresponds to the patient’s right. Example diagram is shown in below. The rotation and flip operation are in Menu "Image>Transform>Rotate".

![image](https://user-images.githubusercontent.com/31482058/117398854-31aecc80-aecd-11eb-9feb-45df93145cd4.png)

2.2 Scroll through all the CT slices in the image file, and write down the starting slice number and the ending slice number that corresponds to the top and bottom boundaries of the patient’s lung.

![image](https://user-images.githubusercontent.com/31482058/116593201-d68a4200-a8ee-11eb-833a-8f9ce9c787b4.png)

**3. Export Lung CT slices**

3.1 Git clone the DECT repository.

3.2. Within this folder in “~/DECT/enhancement_model/Images/original_data”, further create subfolders (naming them for example “CT01”, “CT02”, ...). These subfolders will be used to hold the prepared CT slice images. Each subfolder holds 2D slices of one CT scan.

3.2.1. Export all the CT slices in the image file, by clicking “File” -> “Save As” -> “Image Sequence”. Make sure the “Format” is selected as “TIFF”, “Start At” is 1, and check the “Use slice labels as file names”. Click the “Browse” button to select the output directory (exp: "CT01").

3.2.2. Click “OK”.

3.3. The above step will export all CT slices.

## Next Step
The next step is [2D image enhancement](https://github.com/vtsynergy/2D-DECT) or [3D image enhancement](https://github.com/garvit217-vt/3D-DECT).
