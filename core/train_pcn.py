# -*- coding: utf-8 -*-
# @Author: XP

import logging
import os
import torch
import utils.data_loaders
import utils.helpers
from datetime import datetime
from tqdm import tqdm
from time import time
from tensorboardX import SummaryWriter
from core.test_pcn import test_net
from utils.average_meter import AverageMeter
from models.model import PMPNetPlus as Model
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()

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


def chamfer(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


def lr_lambda(epoch):
    if 0 <= epoch <= 100:
        return 1
    elif 100 < epoch <= 150:
        return 0.5
    elif 150 < epoch <= 250:
        return 0.1
    else:
        return 0.5


def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN),
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    collate_fn=utils.data_loaders.collate_fn,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
                                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                                  num_workers=cfg.CONST.NUM_WORKERS//2,
                                                  collate_fn=utils.data_loaders.collate_fn,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up folders for logs and checkpoints
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', datetime.now().isoformat())
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    model = Model(dataset=cfg.DATASET.TRAIN_DATASET)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # Create the optimizers
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                       lr=cfg.TRAIN.LEARNING_RATE,
                                       weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                       betas=cfg.TRAIN.BETAS)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=lr_lambda)

    init_epoch = 0
    best_metrics = float('inf')

    if 'WEIGHTS' in cfg.CONST and cfg.CONST.WEIGHTS:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        best_metrics = checkpoint['best_metrics']
        model.load_state_dict(checkpoint['model'])
        logging.info('Recover complete. Current epoch = #%d; best metrics = %s.' % (init_epoch, best_metrics))

    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        model.train()

        total_cd1 = 0
        total_cd2 = 0
        total_cd3 = 0
        total_pmd = 0

        batch_end_time = time()
        n_batches = len(train_data_loader)
        with tqdm(train_data_loader) as t:
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(t):
                data_time.update(time() - batch_end_time)
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                partial = random_subsample(data['partial_cloud'])
                gt = random_subsample(data['gtcloud'])

                pcds, deltas = model(partial)

                cd1 = chamfer(pcds[0], gt)
                cd2 = chamfer(pcds[1], gt)
                cd3 = chamfer(pcds[2], gt)
                loss_cd = cd1 + cd2 + cd3

                delta_losses = []
                for delta in deltas:
                    delta_losses.append(torch.sum(delta ** 2))

                loss_pmd = torch.sum(torch.stack(delta_losses)) / 3

                loss = loss_cd * cfg.TRAIN.LAMBDA_CD + loss_pmd * cfg.TRAIN.LAMBDA_PMD

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                cd1_item = cd1.item() * 1e3
                total_cd1 += cd1_item
                cd2_item = cd2.item() * 1e3
                total_cd2 += cd2_item
                cd3_item = cd3.item() * 1e3
                total_cd3 += cd3_item
                pmd_item = loss_pmd.item()
                total_pmd += pmd_item
                n_itr = (epoch_idx - 1) * n_batches + batch_idx
                train_writer.add_scalar('Loss/Batch/cd1', cd1_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd2', cd2_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd3', cd3_item, n_itr)
                train_writer.add_scalar('Loss/Batch/pmd', pmd_item, n_itr)
                batch_time.update(time() - batch_end_time)
                batch_end_time = time()
                t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches))
                t.set_postfix(loss='%s' % ['%.4f' % l for l in [cd1_item, cd2_item, cd3_item, pmd_item]])

        avg_cd1 = total_cd1 / n_batches
        avg_cd2 = total_cd2 / n_batches
        avg_cd3 = total_cd3 / n_batches
        avg_pmd = total_pmd / n_batches

        lr_scheduler.step()
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch/cd1', avg_cd1, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd2', avg_cd2, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd3', avg_cd3, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/pmd', avg_pmd, epoch_idx)
        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cd1, avg_cd2, avg_cd3, avg_pmd]]))

        # Validate the current model
        cd_eval = test_net(cfg, epoch_idx, val_data_loader, val_writer, model)

        # Save checkpoints
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or cd_eval < best_metrics:
            file_name = 'ckpt-best.pth' if cd_eval < best_metrics else 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': best_metrics,
                'model': model.state_dict()
            }, output_path)

            logging.info('Saved checkpoint to %s ...' % output_path)
            if cd_eval < best_metrics:
                best_metrics = cd_eval

    train_writer.close()
    val_writer.close()
