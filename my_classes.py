import numpy as np
import pandas as pd
from PIL import Image
import albumentations as aug



# classes for data loading and preprocessing
class Dataset:
    """Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): text file containing a list of paths to images dir
        masks_dir (str): text file containing a list of paths to segmentation masks

        cats_replace (dict): values of classes to replace in segmentation mask

    """

    cats_replace = {0: [0, 1, 2, 3, 4, 5, 6],
     1: [7, 8, 9, 10],
     2: [11, 12, 13, 14, 15, 16],
     3: [17, 18, 19, 20],
     4: [21, 22],
     5: [23],
     6: [24, 25],
     7: [26, 27, 28, 29, 30, 31, 32, 33, -1]
    }

    DATA_PATH = './data/'

    def __init__(
            self,
            images_dir,
            masks_dir
    ):
        with open(self.DATA_PATH + images_dir, 'r') as f:
            self.images_fps = [line.strip() for line in f.readlines()]

        with open(self.DATA_PATH + masks_dir, 'r') as f:
            self.masks_fps = [line.strip() for line in f.readlines()]

    def __getitem__(self, i):

        # read data
        #image = cv2.imread(self.images_fps[i])
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #mask = cv2.imread(self.masks_fps[i], 0)

        image = np.asarray(Image.open(self.images_fps[i]))
        mask = np.asarray(Image.open(self.masks_fps[i])) # cv2.imread(self.masks_fps[i], 0)

        # convert mask
        mask = self._convert_mask(mask)

        return image, mask.astype(np.float32)

    def _convert_mask(self, ids_img):
        mask_labelids = pd.DataFrame(ids_img)
        for new_value, old_value in self.cats_replace.items():
            mask_labelids = mask_labelids.replace(old_value, new_value);
        mask_labelids = mask_labelids.to_numpy()

        clc = 8

        msk = np.zeros((mask_labelids.shape[0],mask_labelids.shape[1],clc))
        for li in np.unique(mask_labelids):
            msk[:,:,li] = np.logical_or(msk[:,:,li],(mask_labelids==li))
        return np.array(msk, dtype='uint8')

    def __len__(self):
        return len(self.images_fps)
