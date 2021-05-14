import tif_to_nii

import os
import glob
from multiprocessing import Pool

def tiff2nii1(add):
    output = root + '/Images/HU_enhanced_nii/'
    tif_to_nii.main(add,output)

def tiff2nii2(add):
    output = root + '/Images/HU_original_nii/'
    tif_to_nii.main(add,output)

def __main__(add):
    global root
    root = add
    pool = Pool(18)

    path = os.path.join(root+'/Images/', 'HU_enhanced_nii')
    os.mkdir(path)
    input_list = glob.glob(root + '/Images/HU_enhanced/*/')
    pool.map(tiff2nii1, input_list)

    path = os.path.join(root+'/Images/', 'HU_original_nii')
    os.mkdir(path)
    input_list = glob.glob(root + '/Images/HU_original/*/')
    pool.map(tiff2nii2, input_list)
