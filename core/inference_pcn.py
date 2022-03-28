# -*- coding: utf-8 -*-
# @Author: XP

import logging
import os
import torch
import utils.data_loaders
import utils.helpers
import utils.io
from tqdm import tqdm
from models.model import PMPNetPlus as Model


def random_subsample(pcd, n_points=2048):
    """
    Args:
        pcd: (B, N, 3)

    returns:
        new_pcd: (B, n_points, 3)
    """
    b, n, _ = pcd.shape
    device = pcd.device
    batch_idx = torch.arange(b, dtype=torch.long, device=device).reshape((-1, 1)).repeat(1, n_points)
    idx = torch.cat([torch.randperm(n, dtype=torch.long, device=device)[:n_points].reshape((1, -1)) for i in range(b)], 0)
    return pcd[batch_idx, idx, :]

def inference_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
                                                   batch_size=1,
                                                   num_workers=cfg.CONST.NUM_WORKERS,
                                                   collate_fn=utils.data_loaders.collate_fn,
                                                   pin_memory=True,
                                                   shuffle=False)

    model = Model(dataset=cfg.DATASET.TRAIN_DATASET)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    assert 'WEIGHTS' in cfg.CONST and cfg.CONST.WEIGHTS
    # Load the pretrained model from a checkpoint
    logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    model.load_state_dict(checkpoint['model'])

    # Switch models to evaluation mode
    model.eval()

    # The inference loop
    n_samples = len(test_data_loader)
    t_obj = tqdm(test_data_loader)


    for model_idx, (taxonomy_id, model_id, data) in enumerate(t_obj):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        model_id = model_id[0]

        with torch.no_grad():
            for k, v in data.items():
                data[k] = utils.helpers.var_or_cuda(v)

            partial = data['partial_cloud']
            partial = random_subsample(partial.repeat((1, 8, 1)).reshape(-1, 16384, 3))  # b*8, 2048, 3
            pcds = model(partial)[0]

            pcd1 = pcds[0].reshape(-1, 16384, 3).contiguous()
            pcd2 = pcds[1].reshape(-1, 16384, 3).contiguous()
            pcd3 = pcds[2].reshape(-1, 16384, 3).contiguous()

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

            output_file_path = os.path.join(output_folder, 'pcd1', '%s.h5' % model_id)
            utils.io.IO.put(output_file_path, pcd1.squeeze().cpu().numpy())

            output_file_path = os.path.join(output_folder, 'pcd2', '%s.h5' % model_id)
            utils.io.IO.put(output_file_path, pcd2.squeeze().cpu().numpy())

            output_file_path = os.path.join(output_folder, 'pcd3', '%s.h5' % model_id)
            utils.io.IO.put(output_file_path, pcd3.squeeze().cpu().numpy())

            t_obj.set_description('Test[%d/%d] Taxonomy = %s Sample = %s File = %s' %
                         (model_idx + 1, n_samples, taxonomy_id, model_id, output_file_path))

