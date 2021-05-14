import os
import glob
from shutil import copy
from multiprocessing import Pool

def sp_tiff(add):
    # test if enough images
    images = glob.glob(add+'/*.tif')
    if len(images) > 100:
        # new folder
        name_lists = add.split('/')
        folder_name = name_lists[len(name_lists)-1]
        folder_path = root + '/Images/Tiff_in_stacks/'
        path = os.path.join(folder_path, folder_name)
        if not os.path.exists(path):
            os.mkdir(path)


        # copy images
        for i in images:
            copy(i, folder_path+folder_name)

def __main__(add):
    global root
    root = add
    input_list = glob.glob(root+'/Images/original_data/*')

    pool = Pool(18)
    pool.map(sp_tiff, input_list)
