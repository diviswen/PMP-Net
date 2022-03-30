from config_pcn import cfg
from tqdm import tqdm
import utils.data_loaders_jt as dataloader_jt

if __name__ == "__main__":
    train_dataset_loader = dataloader_jt.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    loader = train_dataset_loader.get_dataset(dataloader_jt.DatasetSubset.TRAIN, batch_size=24, shuffle=False)
    for i, o in enumerate(tqdm(loader)):
        loader.display_worker_status()
        # print(len(o[2]['partial_cloud'].shape))