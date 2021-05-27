# PMP-Net

This repository contains the source code for the paper [PMP-Net: Point Cloud Completion by Learning Multi-step Point Moving Paths](https://arxiv.org/abs/2012.03408).
[<img src="pics/network.png" width="100%" alt="Intro pic" />](pics/network.png)

## Cite this work

```
@inproceedings{wen2021pmp,
  title={PMP-Net: Point cloud completion by learning multi-step point moving paths},
  author={Wen, Xin and Xiang, Peng and Han, Zhizhong and Cao, Yan-Pei and Wan, Pengfei and Zheng, Wen and Liu, Yu-Shen},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```

## Datasets

We use the [PCN](https://www.shapenet.org/) and [Compeletion3D](http://completion3d.stanford.edu/) datasets in our experiments, which are available below:

- [PCN](https://drive.google.com/drive/folders/1P_W1tz5Q4ZLapUifuOE4rFAZp6L1XTJz)
- [Completion3D](http://download.cs.stanford.edu/downloads/completion3d/dataset2019.zip)

The pretrained models on Completion3D and PCN dataset are available as follows:

- [PMP-Net_pre-trained](https://drive.google.com/drive/folders/1emGsfdnIj1eUtUxZlfiWiuJ0QJag4nOn?usp=sharing)

Backup Links:

- [PMP-Net_pre-trained](https://pan.baidu.com/s/1oQbaVI7yN9NmI_2E9tztGQ) (pwd: n7t4)

#### Install Python Denpendencies

```
cd PMP-Net
pip install -r requirements.txt
```

#### Build PyTorch Extensions

**NOTE:** PyTorch >= 1.4 of cuda version are required.

```
cd pointnet2_ops_lib
python setup.py install

cd ..

cd Chamfer3D
python setup.py install
```

You need to update the file path of the datasets:

```
__C.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH    = '/path/to/datasets/Completion3D/%s/partial/%s/%s.h5'
__C.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH   = '/path/to/datasets/Completion3D/%s/gt/%s/%s.h5'
__C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH        = '/path/to/datasets/ShapeNet/ShapeNetCompletion/%s/partial/%s/%s/%02d.pcd'
__C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH       = '/path/to/datasets/ShapeNet/ShapeNetCompletion/%s/complete/%s/%s.pcd'

# Dataset Options: Completion3D, Completion3DPCCT, ShapeNet, ShapeNetCars
__C.DATASET.TRAIN_DATASET                        = 'ShapeNet'
__C.DATASET.TEST_DATASET                         = 'ShapeNet'
```

## Getting Started

To train PMP-Net, you can simply use the following command:

```
python main_*.py  # remember to change '*' to 'c3d' or 'pcn'
```

To test or inference, you should specify the path of checkpoint if the config_*.py file
```
__C.CONST.WEIGHTS                                = "path to your checkpoint"
```

then use the following command:

```
python main_*.py --test
python main_*.py --inference
```

## License

This project is open sourced under MIT license.
