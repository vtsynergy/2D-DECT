import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from glob import glob
from PIL import Image

def seperatPath(filepath):
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext


def main(input, output):
    try:
        img_dir = Path(input)
        fns = sorted([str(fn) for fn in img_dir.glob('*.tif*')])
        if not fns:
            raise ValueError(f'img_dir ({input}) does not contain any .tif or .tiff images.')
        imgs = []
        for fn in fns:
            _, base, ext = seperatPath(fn)
            img = np.asarray(Image.open(fn)).astype(np.float32).squeeze()
            # img = np.asarray(Image.open(fn)).astype(np.uint16).squeeze()
            # #####
            # img = img[:,:,0]
            # #####
            if img.ndim != 2:
                raise Exception(f'Only 2D data supported. File {base}{ext} has dimension {img.ndim}.')
            imgs.append(img)
        img = np.stack(imgs, axis=2)
        path_lists = input.split('/')
        nii_name = path_lists[len(path_lists)-2]
        nib.Nifti1Image(img,None).to_filename(os.path.join(output, nii_name+'.nii.gz'))
        return 0
    except Exception as e:
        print(e)
        return 1

if __name__ == "__main__":
    sys.exit(main())
