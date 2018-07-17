import numpy as np
import glob
import random
import cv2
from torch.utils.data import *
from torchvision import transforms
import torch


############################################################################
#  Loader utilities
############################################################################

class NormDenorm:
    # Store mean and std for transforms, apply normalization and de-normalization
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def norm(self, img):
        # normalize image to feed to network
        return img * self.std + self.mean

    def denorm(self, img, cpu=True, variable=True):
        # reverse normalization for viewing
        if cpu:
            img = img.cpu()
        if variable:
            img = img.data

        img = img.numpy().transpose(1, 2, 0)
        return (img - self.mean) / self.std


def cv2_open(fn):
    # Get image with cv2 and convert from bgr to rgb
    try:
        im = cv2.imread(str(fn), cv2.IMREAD_UNCHANGED + cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR).astype(
            np.float32) / 255
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f'Image Open Failure:{fn}  Error:{e}')


############################################################################
#  Image augmentation
############################################################################

def make_img_square(input_img):
    # Take rectangular image and crop to squre
    height = input_img.shape[0]
    width = input_img.shape[1]
    if height > width:
        input_img = input_img[height // 2 - (width // 2):height // 2 + (width // 2), :, :]
    if width > height:
        input_img = input_img[:, width // 2 - (height // 2):width // 2 + (height // 2), :]
    return input_img


class FlipCV(object):
    # resize image and bbox
    def __init__(self, p_x=.5, p_y=.5):
        self.p_x = p_x
        self.p_y = p_y

    def __call__(self, sample):

        flip_x = self.p_x > random.random()
        flip_y = self.p_y > random.random()
        if not flip_x and not flip_y:
            return sample
        else:
            image = sample['image']
            if flip_x and not flip_y:
                image = cv2.flip(image, 1)
            if flip_y and not flip_x:
                image = cv2.flip(image, 0)
            if flip_x and flip_y:
                image = cv2.flip(image, -1)
            return {'image': image}


class ResizeCV(object):
    # resize image and bbox
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        image = make_img_square(image)
        image = cv2.resize(image, (self.output_size, self.output_size), interpolation=cv2.INTER_AREA)
        return {'image': image}


###########################################################################
#  Dataset and Loader
############################################################################
#

class RandomImageDataset(Dataset):
    # Load Image and Apply Augmentation
    def get_subset(self, perc, seed=5):
        # use to only load a percentage of the training set
        total_count = len(self.path_list_a)
        ids = list(range(total_count))

        # seed this consistently even if model restarted
        np.random.seed(seed)

        ids = np.random.permutation(ids)
        split_index = int(total_count * perc)
        subset_ids = ids[:split_index]
        return subset_ids

    def __init__(self, path_a, transform, output_res=64, perc=.1):
        self.transform = transform
        self.path_list_a = sorted(glob.glob(f'{path_a}/*.*'))
        self.ids = self.get_subset(perc)
        self.output_res = output_res

    def transform_set(self, image):
        # Apply augmentation
        trans_dict = {'image': image}
        data_transforms = transforms.Compose([ResizeCV(self.output_res), FlipCV(p_x=.5, p_y=0)])

        trans_dict = data_transforms(trans_dict)
        return np.rollaxis(self.transform.norm(trans_dict['image']), 2)

    def __getitem__(self, index):
        lookup_id = self.ids[index]
        image_path = self.path_list_a[lookup_id]
        image = cv2_open(image_path)
        image = self.transform_set(image)

        tensor = torch.FloatTensor(image)
        return tensor

    def __len__(self):
        return self.ids.size


def data_load(path_a, transform, batch_size, shuffle=False, output_res=128, perc=.1):
    # Wrapper for loader
    dataset = RandomImageDataset(path_a, transform, output_res=output_res, perc=perc)
    datalen = dataset.__len__()
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=shuffle), datalen
