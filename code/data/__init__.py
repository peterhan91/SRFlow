# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains content licensed by https://github.com/xinntao/BasicSR/blob/master/LICENSE/LICENSE

'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data
import numpy as np


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt.get('phase', 'test')
    if phase == 'train':
        gpu_ids = opt.get('gpu_ids', None)
        gpu_ids = gpu_ids if gpu_ids else []
        num_workers = dataset_opt['n_workers'] * len(gpu_ids)
        batch_size = dataset_opt['batch_size']
        shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode'] # mode ~ which dataset to use
    if mode == 'FastMRI':
        from data.fastmri_dataset import FASTMRIDataset as D
        from data.fastmri import subsample, transforms
        # Create a mask function
        mask_func = subsample.HammingMaskFunc(
            # center_fractions=[0.08],
            accelerations=[dataset_opt['factor']]
        )
        class DataTransform:
            def __call__(self, target, mask_func, seed=None):
                # Preprocess the data here
                # target shape: [H, W, 1] or [H, W, 3]
                img = target
                if target.shape[2] != 2:
                    img = np.concatenate((target, np.zeros_like(target)), axis=2)
                assert img.shape[-1] == 2
                img = transforms.to_tensor(img)
                kspace = transforms.fft2(img) 

                center_kspace, _ = transforms.apply_mask(kspace, mask_func, hamming=True, seed=seed)
                img_LF = transforms.complex_abs(transforms.ifft2(center_kspace))
                img_LF = img_LF.unsqueeze(0)
                # img_LF tensor should have shape [H, W, ?]
                target = transforms.to_tensor(np.transpose(target, (2, 0, 1)))  # target shape [1, H, W]
                return img_LF, target
        dataset = D(dataset_opt, mask_func, transform=DataTransform())
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
