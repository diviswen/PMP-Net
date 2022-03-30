# -*- coding: utf-8 -*-
# @Author: XP

import logging
import os
import jittor
import utils.helpers
import utils.io
import utils.data_loaders as dataloader_jt
from tqdm import tqdm
from models.model import PMPNetPlus as Model


def inference_net(cfg):
    dataset_loader = dataloader_jt.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    test_data_loader = dataset_loader.get_dataset(dataloader_jt.DatasetSubset.TEST,
                                                  batch_size=1,
                                                  shuffle=False)

    model = Model(dataset=cfg.DATASET.TEST_DATASET)

    assert 'WEIGHTS' in cfg.CONST and cfg.CONST.WEIGHTS
    print('loading: ', cfg.CONST.WEIGHTS)
    model.load(cfg.CONST.WEIGHTS)

    # Switch models to evaluation mode
    model.eval()

    # The inference loop
    n_samples = len(test_data_loader)
    t_obj = tqdm(test_data_loader)


    for model_idx, (taxonomy_id, model_id, data) in enumerate(t_obj):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        model_id = model_id[0]


        partial = jittor.array(data['partial_cloud'])

        pcds = model(partial)[0]
        pcd1, pcd2, pcd3 = pcds


        output_folder = os.path.join(cfg.DIR.OUT_PATH, 'benchmark', taxonomy_id)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_folder_pcd1 = os.path.join(output_folder, 'pcd1')
        output_folder_pcd2 = os.path.join(output_folder, 'pcd2')
        output_folder_pcd3 = os.path.join(output_folder, 'pcd3')
        if not os.path.exists(output_folder_pcd1):
            os.makedirs(output_folder_pcd1)
            os.makedirs(output_folder_pcd2)
            os.makedirs(output_folder_pcd3)

        # print(pcd1)
        output_file_path = os.path.join(output_folder, 'pcd1', '%s.h5' % model_id)
        utils.io.IO.put(output_file_path, pcd3.squeeze(0).detach().numpy())

        output_file_path = os.path.join(output_folder, 'pcd2', '%s.h5' % model_id)
        utils.io.IO.put(output_file_path, pcd2.squeeze(0).detach().numpy())

        output_file_path = os.path.join(output_folder, 'pcd3', '%s.h5' % model_id)
        utils.io.IO.put(output_file_path, pcd3.squeeze(0).detach().numpy())

        t_obj.set_description('Test[%d/%d] Taxonomy = %s Sample = %s File = %s' %
                     (model_idx + 1, n_samples, taxonomy_id, model_id, output_file_path))



