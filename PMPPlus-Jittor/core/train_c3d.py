# -*- coding: utf-8 -*-
# @Author: XP

import logging
import os
import jittor
import utils.data_loaders as dataloader_jt
from jittor import nn
from datetime import datetime
from tqdm import tqdm
from time import time
from tensorboardX import SummaryWriter
from core.test_c3d import test_net
from utils.average_meter import AverageMeter
from models.model import PMPNetPlus as Model
from core.chamfer import chamfer_loss_bidirectional as chamfer
from jittor.utils.nvtx import nvtx_scope

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

    # train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    # test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)

    train_dataset_loader = dataloader_jt.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    test_dataset_loader = dataloader_jt.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)

    train_data_loader = train_dataset_loader.get_dataset(dataloader_jt.DatasetSubset.TRAIN,
                                                         batch_size=cfg.TRAIN.BATCH_SIZE,
                                                         num_workers=cfg.CONST.NUM_WORKERS,
                                                         shuffle=True)
    val_data_loader = test_dataset_loader.get_dataset(dataloader_jt.DatasetSubset.VAL,
                                                      batch_size=cfg.TRAIN.BATCH_SIZE,
                                                      num_workers=cfg.CONST.NUM_WORKERS,
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
    init_epoch = 0
    best_metrics = float('inf')

    optimizer = nn.Adam(model.parameters(),
                        lr=cfg.TRAIN.LEARNING_RATE,
                        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                        betas=cfg.TRAIN.BETAS)
    lr_scheduler = jittor.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=cfg.TRAIN.LR_MILESTONES,
                                                   gamma=cfg.TRAIN.GAMMA,
                                                   last_epoch=init_epoch)



    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()
        # cd_eval = test_net(cfg, epoch_idx, val_data_loader, val_writer, model)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        # cd_eval = test_net(cfg, epoch_idx, val_data_loader, val_writer, model)
        model.train()

        total_cd1 = 0
        total_cd2 = 0
        total_cd3 = 0
        total_pmd = 0
        batch_end_time = time()
        n_batches = len(train_data_loader)
        print('epoch: ', epoch_idx, 'optimizer: ', lr_scheduler.get_lr())
        with tqdm(train_data_loader) as t:
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(t):
                with nvtx_scope("model"):
                    partial = jittor.array(data['partial_cloud'])
                    gt = jittor.array(data['gtcloud'])
                    pcds, deltas = model(partial)

                    cd1 = chamfer(pcds[0], gt)
                    cd2 = chamfer(pcds[1], gt)
                    cd3 = chamfer(pcds[2], gt)
                    loss_cd = cd1 + cd2 + cd3

                    delta_losses = []
                    for delta in deltas:
                        delta_losses.append(jittor.sum(delta ** 2))

                    loss_pmd = jittor.sum(jittor.stack(delta_losses)) / 3

                    loss = loss_cd * cfg.TRAIN.LAMBDA_CD + loss_pmd * cfg.TRAIN.LAMBDA_PMD

                    cd1_item = cd1.item() * 1e3
                    total_cd1 += cd1_item
                    cd2_item = cd2.item() * 1e3
                    total_cd2 += cd2_item
                    cd3_item = cd3.item() * 1e3
                    total_cd3 += cd3_item
                    pmd_item = loss_pmd.item()
                    total_pmd += pmd_item

                with nvtx_scope("step"):
                    optimizer.step(loss)
                with nvtx_scope("sync_all"):
                    jittor.sync_all()

                n_itr = (epoch_idx - 1) * n_batches + batch_idx
                train_writer.add_scalar('Loss/Batch/cd1', cd1_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd2', cd2_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd3', cd3_item, n_itr)
                train_writer.add_scalar('Loss/Batch/pmd', pmd_item, n_itr)
                batch_time.update(time() - batch_end_time)
                batch_end_time = time()
                t.set_description(
                    '[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches))
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
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time,
             ['%.4f' % l for l in [avg_cd1, avg_cd2, avg_cd3, avg_pmd]]))

        # Validate the current model
        cd_eval = test_net(cfg, epoch_idx, val_data_loader, val_writer, model)

        # Save checkpoints
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or cd_eval < best_metrics:
            file_name = 'ckpt-best.pkl' if cd_eval < best_metrics else 'ckpt-epoch-%03d.pkl' % epoch_idx
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)

            model.save(output_path)

            logging.info('Saved checkpoint to %s ...' % output_path)
            if cd_eval < best_metrics:
                best_metrics = cd_eval

    train_writer.close()
    val_writer.close()
