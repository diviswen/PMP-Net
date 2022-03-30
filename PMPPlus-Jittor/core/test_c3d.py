# -*- coding: utf-8 -*-
# @Author: XP

import jittor
import utils.data_loaders as dataloader_jt
from tqdm import tqdm
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
from models.model import PMPNetPlus as Model
from core.chamfer import chamfer_loss_bidirectional, chamfer_loss_bidirectional_sqrt
chamfer = chamfer_loss_bidirectional
chamfer_sqrt = chamfer_loss_bidirectional_sqrt

def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, model=None):

    if test_data_loader is None:
        # Set up data loader
        dataset_loader = dataloader_jt.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = dataset_loader.get_dataset(dataloader_jt.DatasetSubset.VAL,
                                                         batch_size=1,
                                                         shuffle=False)


    # Setup networks and initialize networks
    if model is None:
        model = Model(dataset=cfg.DATASET.TEST_DATASET)

        assert 'WEIGHTS' in cfg.CONST and cfg.CONST.WEIGHTS
        print('loading: ', cfg.CONST.WEIGHTS)
        model.load(cfg.CONST.WEIGHTS)

    # Switch models to evaluation mode
    model.eval()

    n_samples = len(test_data_loader)
    test_losses = AverageMeter(['cd1', 'cd2', 'cd3', 'pmd'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()

    # Testing loop
    with tqdm(test_data_loader) as t:
        # print('repeating')
        for model_idx, (taxonomy_id, model_id, data) in enumerate(t):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            model_id = model_id[0]


            # for k, v in data.items():
                # data[k] = utils.helpers.var_or_cuda(v)

            partial = jittor.array(data['partial_cloud'])
            gt = jittor.array(data['gtcloud'])

            b, n, _ = partial.shape

            pcds, deltas = model(partial)

            cd1 = chamfer(pcds[0], gt).item() * 1e3
            cd2 = chamfer(pcds[1], gt).item() * 1e3
            cd3 = chamfer(pcds[2], gt).item() * 1e3

            # pmd loss
            pmd_losses = []
            for delta in deltas:
                pmd_losses.append(jittor.sum(delta ** 2))


            pmd = jittor.sum(jittor.stack(pmd_losses)) / 3

            pmd_item = pmd.item()

            _metrics = [pmd_item, cd3]
            test_losses.update([cd1, cd2, cd3, pmd_item])

            test_metrics.update(_metrics)
            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            t.set_description('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                         (model_idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()
                                                                            ], ['%.4f' % m for m in _metrics]))

    # Print testing results
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items:
        print(metric, end='\t')
    print()

    for taxonomy_id in category_metrics:
        print(taxonomy_id, end='\t')
        print(category_metrics[taxonomy_id].count(0), end='\t')
        for value in category_metrics[taxonomy_id].avg():
            print('%.4f' % value, end='\t')
        print()

    print('Overall', end='\t\t\t')
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/cd1', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/cd2', test_losses.avg(1), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/cd3', test_losses.avg(2), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/delta', test_losses.avg(3), epoch_idx)
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)
    model.train()
    return test_losses.avg(2)
