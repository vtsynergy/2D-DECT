import stack_select
import train_main2_jy
import batch_tif2nii
import batch_rename_nii

import os
import glob
from subprocess import Popen
from multiprocessing import Pool
import shutil

# mycwd = os.getcwd()
mycwd = os.path.split(os.path.realpath(__file__))[0]
# print(os.path.split(os.path.realpath(__file__))[0])
print('root is: ',mycwd[:-5])
root = mycwd[:-5]

# 1. select stack with enough images (> 100)
if not os.path.exists(root + '/Images/Tiff_in_stacks'):
    os.mkdir(root + '/Images/Tiff_in_stacks')
stack_select.__main__(root)

# 2. enhance images in 'Tiff_in_stacks' folder
if not os.path.exists(root + '/Images/HU_enhanced'):
    os.mkdir(root + '/Images/HU_enhanced')
if not os.path.exists(root + '/Images/HU_original'):
    os.mkdir(root + '/Images/HU_original')
if not os.path.exists("./visualize"):
    os.makedirs("./visualize")
if not os.path.exists("./visualize"):
    os.makedirs("./visualize/test")
lists = glob.glob(root+'/Images/Tiff_in_stacks/*/')# find all folders
lists.sort()

pool = Pool(7)
pool.map(train_main2_jy.__main__, lists)
shutil.rmtree('./visualize')

# 3. convert TIFF images to NII Images
batch_tif2nii.__main__(root)

# 4. rename NII images
batch_rename_nii.__main__(root)
