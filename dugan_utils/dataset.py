import torch
import torch.utils.data as tordata
import os.path as osp
import numpy as np
from functools import partial


class CTPatchDataset(tordata.Dataset):
    def __init__(self, npy_root, hu_range, transforms=None):
        self.transforms = transforms
        hu_min, hu_max = hu_range
        data = torch.from_numpy(np.load(npy_root).astype(np.float32) - 1024)
        # normalize to [0, 1]
        data = (torch.clamp(data, hu_min, hu_max) - hu_min) / (hu_max - hu_min)
        self.low_doses, self.full_doses = data[0], data[1]

    def __getitem__(self, index):
        low_dose, full_dose = self.low_doses[index], self.full_doses[index]
        if self.transforms is not None:
            low_dose = self.transforms(low_dose)
            full_dose = self.transforms(full_dose)
        return low_dose, full_dose

    def __len__(self):
        return len(self.low_doses)


class AppendixDataset(tordata.Dataset):
    def __init__(self, npy_root, hu_range=None, transforms=None):
        self.transforms = transforms
        # Load data: [2, N, H, W]
        # Assumes data is already normalized to [0, 1] by create_appendix.py
        data = torch.from_numpy(np.load(npy_root).astype(np.float32))
        self.low_doses, self.full_doses = data[0], data[1]

    def __getitem__(self, index):
        low_dose, full_dose = self.low_doses[index], self.full_doses[index]
        # Ensure 3D if needed [1, H, W]
        if low_dose.ndim == 2:
            low_dose = low_dose.unsqueeze(0)
            full_dose = full_dose.unsqueeze(0)
            
        if self.transforms is not None:
            low_dose = self.transforms(low_dose)
            full_dose = self.transforms(full_dose)
        return low_dose, full_dose

    def __len__(self):
        return len(self.low_doses)

def get_combined_dataset(hu_range, transforms=None):
    from torchvision import transforms as T
    
    # CT (Already 64x64)
    ds_ct = CTPatchDataset(npy_root=osp.join(data_root, 'cmayo/train_64.npy'), hu_range=hu_range, transforms=transforms)
    
    # MRI & BUSI (Resize to 64x64 to match CT)
    if transforms is not None:
        combined_transform = T.Compose([T.Resize((64, 64)), transforms])
    else:
        combined_transform = T.Resize((64, 64))

    ds_mri = AppendixDataset(npy_root=osp.join(data_root, 'cmayo_appendix/mri_appendix.npy'), hu_range=hu_range, transforms=combined_transform)
    ds_busi = AppendixDataset(npy_root=osp.join(data_root, 'cmayo_appendix/busi_appendix.npy'), hu_range=hu_range, transforms=combined_transform)
    
    return tordata.ConcatDataset([ds_ct, ds_mri, ds_busi])

data_root = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'dataset')
dataset_dict = {
    'cmayo_train_64': partial(CTPatchDataset, npy_root=osp.join(data_root, 'cmayo/train_64.npy')),
    'cmayo_test_512': partial(CTPatchDataset, npy_root=osp.join(data_root, 'cmayo/test_512.npy')),
    'mri_appendix': partial(AppendixDataset, npy_root=osp.join(data_root, 'cmayo_appendix/mri_appendix.npy')),
    'busi_appendix': partial(AppendixDataset, npy_root=osp.join(data_root, 'cmayo_appendix/busi_appendix.npy')),
    'combined_train': get_combined_dataset,
}
