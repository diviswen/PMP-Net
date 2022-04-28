# PMP-Net++: Point Cloud Completion by Transformer-Enhanced Multi-step Point Moving Paths (TPAMI 2022) (Jittor Implementation)

[<img src="../pics/network.png" width="100%" alt="Intro pic" />](../pics/network.png)


## [PMP-Net++]
This repository contains the [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) implementation of the papers:

**1. PMP-Net++: Point Cloud Completion by Transformer-Enhanced Multi-step Point Moving Paths, TPAMI 2022**

**2. PMP-Net: Point Cloud Completion by Learning Multi-step Point Moving Paths, CVPR 2021**

The **Jittor** implementation on different datasets:
| Model       | Completion3d   | PCN dataset |
| ----------- | -------------- | ------------ |
| PMP-Net     | √              | √            |
| PMP-Net++   | √              | √            |

[ [PMP-Net](https://arxiv.org/abs/2012.03408) | [PMP-Net++](https://arxiv.org/abs/2012.03408) | [IEEE Xplore](https://ieeexplore.ieee.org/document/9735342) | [Webpage]() | [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) ] 

> Point cloud completion concerns to predict missing part for incomplete 3D shapes. A common strategy is to generate
complete shape according to incomplete input. However, unordered nature of point clouds will degrade generation of high-quality 3D
shapes, as detailed topology and structure of unordered points are hard to be captured during the generative process using an
extracted latent code. We address this problem by formulating completion as point cloud deformation process. Specifically, we design a
novel neural network, named PMP-Net++, to mimic behavior of an earth mover. It moves each point of incomplete input to obtain a
complete point cloud, where total distance of point moving paths (PMPs) should be the shortest. Therefore, PMP-Net++ predicts
unique PMP for each point according to constraint of point moving distances. The network learns a strict and unique correspondence
on point-level, and thus improves quality of predicted complete shape. Moreover, since moving points heavily relies on per-point
features learned by network, we further introduce a transformer-enhanced representation learning network, which significantly
improves completion performance of PMP-Net++. We conduct comprehensive experiments in shape completion, and further explore
application on point cloud up-sampling, which demonstrate non-trivial improvement of PMP-Net++ over state-of-the-art point cloud
completion/up-sampling methods

## [Cite this work]

```
@ARTICLE{pmpnet++,
    author={Wen, Xin and Xiang, Peng and Han, Zhizhong and Cao, Yan-Pei and Wan, Pengfei and Zheng, Wen and Liu, Yu-Shen},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
    title={PMP-Net++: Point Cloud Completion by Transformer-Enhanced Multi-step Point Moving Paths}, 
    year={2022},
    volume={},
    number={},
    pages={1-1},
    doi={10.1109/TPAMI.2022.3159003}
}

@inproceedings{wen2021pmp,
    title={PMP-Net: Point cloud completion by learning multi-step point moving paths},
    author={Wen, Xin and Xiang, Peng and Han, Zhizhong and Cao, Yan-Pei and Wan, Pengfei and Zheng, Wen and Liu, Yu-Shen},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2021}
}
```

## [Getting Started]
#### Datasets and Pretrained Models

We use the [PCN](https://www.shapenet.org/) and [Compeletion3D](http://completion3d.stanford.edu/) datasets in our experiments, which are available below:

- [PCN](https://drive.google.com/drive/folders/1P_W1tz5Q4ZLapUifuOE4rFAZp6L1XTJz)
- [Completion3D](http://download.cs.stanford.edu/downloads/completion3d/dataset2019.zip)


#### Install Python Denpendencies

```
conda create -n pmp python=3.7
conda activate pmp
pip3 install -r requirements.txt

python3.7 -m pip install jittor
python3.7 -m jittor.test.test_example
python3.7 -m jittor.test.test_cudnn_op
# more information about Jittor can be found at https://cg.cs.tsinghua.edu.cn/jittor/
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

#### Training, Testing and Inference

To train PMP-Net++ or PMP-Net, you can simply use the following command:

```
python main_*.py  # remember to change '*' to 'c3d' or 'pcn', and change between 'import PMPNetPlus' and 'import PMPNet'
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

## [Acknowledgements]

Some of the code of this repo is borrowed from [GRNet](https://github.com/hzxie/GRNet) and [PointCloudLib](https://github.com/Jittor/PointCloudLib). We thank the authors for their wonderful job!

## [License]

This project is open sourced under MIT license.
