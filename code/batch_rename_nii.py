import os
import glob
from multiprocessing import Pool

def changeName(add):
    lists = add.split('.')
    if(len(lists)>3):
        os.rename(add,lists[0]+'_'+lists[1]+'.'+lists[2]+'.'+lists[3])

def __main__(add):
    root = add
    pool = Pool(19)
    zip_list = glob.glob(root + '/Images/HU_enhanced_nii/*.nii.gz')
    pool.map(changeName, zip_list)
    zip_list = glob.glob(root + '/Images/HU_original_nii/*.nii.gz')
    pool.map(changeName, zip_list)
