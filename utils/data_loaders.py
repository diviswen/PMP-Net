# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:21:32
# @Email:  cshzxie@gmail.com

import json
import logging
import numpy as np
import random
import torch.utils.data.dataset
import h5py as H
import os
import open3d as o3d
import utils.data_transforms
from enum import Enum, unique
from tqdm import tqdm
from utils.io import IO

label_mapping = {
    3: '03001627',
    6: '04379243',
    5: '04256520',
    1: '02933112',
    4: '03636649',
    2: '02958343',
    0: '02691156',
    7: '04530566'
}

@unique
class DatasetSubset(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


def collate_fn(batch):
    taxonomy_ids = []
    model_ids = []
    data = {}

    for sample in batch:
        taxonomy_ids.append(sample[0])
        model_ids.append(sample[1])
        _data = sample[2]
        for k, v in _data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items():
        data[k] = torch.stack(v, 0)

    return taxonomy_ids, model_ids, data

code_mapping = {
    'plane': '02691156',
    'cabinet': '02933112',
    'car': '02958343',
    'chair': '03001627',
    'lamp': '03636649',
    'couch': '04256520',
    'table': '04379243',
    'watercraft': '04530566',
}

def read_ply(file_path):
    pc = o3d.io.read_point_cloud(file_path)
    ptcloud = np.array(pc.points)
    return ptcloud


class UpSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        curr = ptcloud.shape[0]
        need = self.n_points - curr

        if need < 0:
            return ptcloud[np.random.permutation(self.n_points)]

        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            need -= curr
            curr *= 2

        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

        return ptcloud


class MyShapeNetDataSet(torch.utils.data.dataset.Dataset):
    def __init__(self, root='/data1/xp/PCN', phase='train', categories=None):
        assert phase in {'train', 'val', 'test'}
        self.phase = phase
        base_dir = os.path.join(root, phase)
        if categories is None:
            self.taxomony_ids = list(code_mapping.values())
        else:
            taxomony_ids = []
            for c in categories:
                taxomony_ids.append(code_mapping[c])
            self.taxomony_ids = taxomony_ids

        all_taxomony_ids = []
        all_model_ids = []
        all_pcds_partial = []
        all_pcds_gt = []

        for t_id in self.taxomony_ids:
            gt_dir = os.path.join(base_dir, 'complete', t_id)
            partial_dir = os.path.join(base_dir, 'partial', t_id)
            model_ids = os.listdir(partial_dir)
            all_taxomony_ids.extend([t_id for i in range(len(model_ids))])
            all_model_ids.extend(model_ids)
            all_pcds_gt.extend([os.path.join(gt_dir, f) for f in sorted(os.listdir(gt_dir))])
            all_pcds_partial.extend([os.path.join(partial_dir, f) for f in sorted(os.listdir(partial_dir))])

        self.taxomony_ids = all_taxomony_ids
        self.model_ids = all_model_ids
        self.path_partial = all_pcds_partial
        self.path_gt = all_pcds_gt
        self.LEN = len(self.model_ids)
        self.transform = UpSamplePoints({'n_points': 2048})

    def __len__(self):
        return len(self.model_ids)

    def __getitem__(self, index):
        if self.phase == 'test':
            partial = read_ply(self.path_partial[index]).astype(np.float32)
        else:
            idx_partial = random.randint(0, 7)
            partial = read_ply(os.path.join(self.path_partial[index], '0{}.pcd'.format(idx_partial))).astype(np.float32)
        partial = self.transform(partial)
        gt = read_ply(self.path_gt[index]).astype(np.float32)
        idx_random_complete = random.randint(0, self.LEN - 1)
        random_complete = read_ply(self.path_gt[idx_random_complete]).astype(np.float32)
        data = {
            'X': torch.from_numpy(partial).float(),
            'Y': torch.from_numpy(random_complete).float(),
            'X_GT': torch.from_numpy(gt).float()
        }
        return self.taxomony_ids[index], self.model_ids[index], data








class CascadeDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, phase='train'):
        base_dir = '/data1/xp/cascaded'
        if phase in {'train', 'valid', 'test'}:
            with H.File(os.path.join(base_dir, '{}_data.h5'.format(phase)), 'r') as f:
                self.gt = f['complete_pcds'][:]
                self.partial = f['incomplete_pcds'][:]
                self.labels = f['labels'][:]
                self.classes = f['classes'][:]

        self.transform = self._get_transforms(phase)

    def _get_transforms(self, subset):
        if subset == 'train':
            return utils.data_transforms.Compose([{
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):

        data = {'partial_cloud': self.partial[idx], 'gtcloud': self.gt[idx]}
        data = self.transform(data)
        return label_mapping[self.labels[idx]], self.classes[idx].astype(str), data


def load_completion3d_data(phase='train'):
    path = '/data1/xp/shapenet'
    if phase in {'train', 'val'}:
        codes = ['02691156', '02958343', '03636649', '03001627', '04379243', '02933112', '04530566', '04256520']
        code_len = 8
    else:
        codes = ['all']
        code_len = 3

    partial_d = {}
    gt_d = {}
    model_id_d = {}
    for code in codes:
        partial_d[code] = []
        gt_d[code] = []
        model_id_d[code] = []

    with open(os.path.join(path, '{}.list'.format(phase)), 'r') as f:
        file_list = f.readlines()
    print('loading phase {}...'.format(phase))
    tqdm_obj = tqdm(file_list)
    for f in tqdm_obj:
        encode = f[:code_len]
        model_id = f[code_len+1:-1]
        filename = model_id + '.h5'
        pcd_file = H.File(os.path.join(path, phase, 'partial', encode, filename), 'r')
        pcd = np.array(pcd_file['data'])
        pcd_file.close()
        partial_d[encode].append(pcd.astype(np.float64))
        pcd_file = H.File(os.path.join(path, phase, 'gt', encode, filename), 'r')
        pcd = np.array(pcd_file['data'])
        pcd_file.close()
        gt_d[encode].append(pcd.astype(np.float64))
        model_id_d[encode].append(model_id)

        # tqdm_obj.set_description('encode {} model_id {}'.format(encode, model_id))
    print('loaded phase {}!'.format(phase))
    if phase == 'train':
        for code in codes:
            n = len(partial_d[code]) % 16
            if n > 0:
                partial_d[code] = partial_d[code][:-n]
                gt_d[code] = gt_d[code][:-n]
                model_id_d[code] = model_id_d[code][:-n]

    partial, gt, model_id, taximony_id = [], [], [], []
    for code in codes:
        partial.append(np.array(partial_d[code]))
        gt.append(np.array(gt_d[code]))
        model_id.extend(model_id_d[code])
        taximony_id.extend([code for i in range(len(partial_d[code]))])

    partial = np.concatenate(partial)
    gt = np.concatenate(gt)

    return partial, gt, model_id, taximony_id

'''
def load_completion3d_data_of_type(path, encode, phase='train'):
    train_list = open(os.path.join(path, '{}.list'.format(phase)), 'r')
    partial = []
    gt = []
    model_id_list = []
    print('loading phase {}, code {}'.format(phase, encode))
    for f in train_list:
        if f.startswith(encode):
            model_id = f[len(encode)+1:-1]
            model_id_list.append(model_id)
            filename = model_id + '.h5'
            pcd_file = H.File(os.path.join(path, phase, 'partial', encode, filename), 'r')
            pcd = np.array(pcd_file['data'])
            pcd_file.close()
            partial.append(pcd.astype(np.float64))
            pcd_file = H.File(os.path.join(path, phase, 'gt', encode, filename), 'r')
            pcd = np.array(pcd_file['data'])
            pcd_file.close()
            gt.append(pcd.astype(np.float64))

    partial = np.array(partial)
    gt = np.array(gt)

    if phase == 'train':
        n = partial.shape[0] % 16
        if n > 0:
            partial = partial[:-n]
            gt = gt[:-n]

    return partial, gt, model_id_list, [encode for i in range(partial.shape[0])]

def load_completion3d_data(phase='train'):
    path = '/data1/xp/shapenet'
    if phase in {'train', 'val'}:
        codes = ['02691156', '02958343', '03636649', '03001627', '04379243', '02933112', '04530566', '04256520']
    else:
        codes = ['all']

    partial = []
    gt = []
    model_id = []
    type_id = []


    for code in codes:
        partial_, gt_, model_id_, type_id_ = load_completion3d_data_of_type(path, code, phase=phase)
        partial.append(partial_)
        gt.append(gt_)
        model_id.extend(model_id_)
        type_id.extend(type_id_)

    return np.concatenate(partial), np.concatenate(gt), model_id, type_id
'''
def load_displacement():
    dis = []
    print('loading displacement...')
    for i in range(4):
        dis.append(np.load('/data1/xp/shapenet/train_displacement_{}.npy'.format(i)))
    print('displacement loaded...')
    return np.concatenate(dis, 0)



class Completion3DDisplacement(torch.utils.data.dataset.Dataset):
    def __init__(self, phase='train'):
        super(Completion3DDisplacement, self).__init__()

        self.phase = phase
        self.partial, self.gt, self.model_id, self.type_id = load_completion3d_data(phase)
        if self.phase == 'train':
            self.displacement = load_displacement()

        self.transform = self._get_transforms(phase)

    def _get_transforms(self, phase):
        if phase == 'train':
            return utils.data_transforms.Compose([{
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud', 'displacement']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud', 'displacement']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])

    def __len__(self):
        return self.partial.shape[0]

    def __getitem__(self, idx):
        data = {'partial_cloud': self.partial[idx], 'gtcloud': self.gt[idx]}
        if self.phase == 'train':
            data['displacement'] = self.displacement[idx]
        data = self.transform(data)

        return self.type_id[idx], self.model_id[idx], data



class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, options, file_list, transforms=None):
        self.options = options
        self.file_list = file_list
        self.transforms = transforms
        self.cache = dict()

    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = -1
        if 'n_renderings' in self.options:
            rand_idx = random.randint(0, self.options['n_renderings'] - 1) if self.options['shuffle'] else 0

        for ri in self.options['required_items']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            # print(file_path)
            data[ri] = IO.get(file_path).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], data


class ShapeNetDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.SHAPENET.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        n_renderings = self.cfg.DATASETS.SHAPENET.N_RENDERINGS if subset == DatasetSubset.TRAIN else 1
        file_list = self._get_file_list(self.cfg, self._get_subset(subset), n_renderings)
        transforms = self._get_transforms(self.cfg, subset)
        return Dataset({
            'n_renderings': n_renderings,
            'required_items': ['partial_cloud', 'gtcloud'],
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):

                if subset == 'test':

                    gt_path = cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s)
                    file_list.append({'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    'partial_cloud_path': gt_path.replace('complete', 'partial'),
                    'gtcloud_path': gt_path})
                else:
                    file_list.append({
                        'taxonomy_id':
                            dc['taxonomy_id'],
                        'model_id':
                            s,
                        'partial_cloud_path': [
                            cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (subset, dc['taxonomy_id'], s, i)
                            for i in range(n_renderings)
                        ],
                        'gtcloud_path':
                            cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s),
                    })

                    '''
                    gt_path = cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s)
                    file_list.extend([{
                        'taxonomy_id': dc['taxonomy_id'],
                        'model_id': s,
                        'partial_cloud_path': cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (
                        subset, dc['taxonomy_id'], s, i),
                        'gtcloud_path': gt_path
                    } for i in range(n_renderings)])
                    '''




        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


class ShapeNetCarsDataLoader(ShapeNetDataLoader):
    def __init__(self, cfg):
        super(ShapeNetCarsDataLoader, self).__init__(cfg)

        # Remove other categories except cars
        self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']


class Completion3DDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)
        required_items = ['partial_cloud'] if subset == DatasetSubset.TEST else ['partial_cloud', 'gtcloud']
        # required_items = ['partial_cloud', 'gtcloud']
        return Dataset({
            'required_items': required_items,
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': cfg.CONST.N_INPUT_POINTS
                },
                'objects': ['partial_cloud']
            },  {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        elif subset == DatasetSubset.VAL:
            return utils.data_transforms.Compose([{
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'ToTensor',
                'objects': ['partial_cloud']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_cloud_path':
                    cfg.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH % (subset, dc['taxonomy_id'], s),
                    'gtcloud_path':
                    cfg.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s),
                })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


class Completion3DPCCTDataLoader(Completion3DDataLoader):
    """
    Dataset Completion3D containing only plane, car, chair, table
    """
    def __init__(self, cfg):
        super(Completion3DPCCTDataLoader, self).__init__(cfg)

        # Remove other categories except couch, chairs, car, lamps
        cat_set = {'02691156', '03001627', '02958343', '04379243'} # plane, chair, car, table
        # cat_set = {'04256520', '03001627', '02958343', '03636649'}
        self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] in cat_set]


class KittiDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.KITTI.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)
        required_items = ['partial_cloud', 'bounding_box']

        return Dataset({'required_items': required_items, 'shuffle': False}, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        return utils.data_transforms.Compose([{
            'callback': 'NormalizeObjectPose',
            'parameters': {
                'input_keys': {
                    'ptcloud': 'partial_cloud',
                    'bbox': 'bounding_box'
                }
            },
            'objects': ['partial_cloud', 'bounding_box']
        }, {
            'callback': 'RandomSamplePoints',
            'parameters': {
                'n_points': cfg.CONST.N_INPUT_POINTS
            },
            'objects': ['partial_cloud']
        }, {
            'callback': 'ToTensor',
            'objects': ['partial_cloud', 'bounding_box']
        }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    'partial_cloud_path': cfg.DATASETS.KITTI.PARTIAL_POINTS_PATH % s,
                    'bounding_box_path': cfg.DATASETS.KITTI.BOUNDING_BOX_FILE_PATH % s,
                })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


# //////////////////////////////////////////// = Dataset Loader Mapping = //////////////////////////////////////////// #

DATASET_LOADER_MAPPING = {
    'Completion3D': Completion3DDataLoader,
    'Completion3DPCCT': Completion3DPCCTDataLoader,
    'ShapeNet': ShapeNetDataLoader,
    'ShapeNetCars': ShapeNetCarsDataLoader,
    'KITTI': KittiDataLoader
}  # yapf: disable

