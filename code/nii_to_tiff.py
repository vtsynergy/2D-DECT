import os
from os import path
import numpy as np
from PIL import Image
import re
from skimage import io, transform
import nibabel as nib
from pathlib import Path
import odl
from pathlib import Path
import matplotlib.pyplot as plt
import util
import glob
import scipy.io
import matplotlib.image as mpimg
import re


data_root = "../"

out_root = "./out/"

if not os.path.exists(out_root):
    os.makedirs(out_root)

inputs_np = None
targets_np = None
input_files = []

folder_name = 1

total_folders = 500
img_count = 0

for path in Path(data_root).rglob('*.nii.gz'):
    p = path.resolve()
    path_nii = str(p)

    nii_img = nib.load(path_nii)

    num_images = nii_img.shape[2]
    print(path_nii)

    #num_discarded_images = num_images % images_per_folder
    #num_images = num_images - num_discarded_images

    folder_name = folder_name + 1
    print("Making folder ", folder_name)
    nii_img = nii_img.get_fdata()
    count = 0

    for i in range(num_images):
        tif_img = nii_img[:, :, i]

        #ct_org = np.int32(tif_img)
        #ct_org = ct_org + 1024

        if((ct_org.shape[0] != 512) or (ct_org.shape[1] != 512)):
            print("removed")
            continue

        neg_val_index = ct_org < -1024
        ct_org[neg_val_index] = -1024

        file_name = "image_" + str(count) + ".tif"
        dest_folder = out_root + "BIMCV_" + str(folder_name) + "/"

        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        
        im = Image.fromarray(ct_org)
        im.save(dest_folder + file_name)

        count = count + 1

        if(folder_name - 1 == total_folders):
            exit()
