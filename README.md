# pedestrian_tracking
Pedestrian Tracking by DeepSORT and Hybrid Task Cascade with PyTorch.

![](https://github.com/FinalFlowers/pedestrian_tracking/blob/master/results/2.png?raw=true)

## Introduction
This project is used to participate in zte algorithm contest（中兴捧月算法大赛阿尔法·勒克斯特派）, which get 77.838 on the A board.

Pedestrian detection is obtained by [Hybrid Task Cascade](https://arxiv.org/abs/1901.07518), which implemented by [MMDetection](https://github.com/open-mmlab/mmdetection). 

I choose to use [DeepSORT](https://arxiv.org/abs/1703.07402) to achieve the data association. This section is modified by other authors' [implementation](https://github.com/ZQPei/deep_sort_pytorch).

Several other detection algorithms, such as [Cascade R-CNN](https://arxiv.org/abs/1712.00726) and [EfficientDet](https://arxiv.org/abs/1911.09070), were also tested, but with poor results.

## Installation
#### 1.Download the project
`git clone https://github.com/FinalFlowers/pedestrian_tracking.git`

#### 2.Install the required libraries
`cd pedestrian_tracking`

`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

#### 3.Compile MMDetection
`pip install -v -e .`

**Note**: there is a point at the end of the command.

#### 4.Download the weight files
Download detection and ReID feature extraction model parameters from [Baidu Netdisk](https://pan.baidu.com/s/1gfRnIcaNJIb2NcOPc2fKgQ) with code: *bboh*.

Put `htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.pth` under `person_tracking/models/`

Put `ckpt.t7` under `person_tracking/deep_sort/deep/checkpoint/`

## Testing
Run the following code for pedestrian tracking:

`python htc_deepsort.py /your/trackdata/`

The output format is:

`<frame>，<id>，<bb_left>，<bb_top>，<bb_width>，<bb_height>，<conf>，<type> `

**Note**:

- `Conf` and `type` are fixed as 0.9 and 0 respectively.

- The input should be a path to images ending in `/`

- The results will be saved under `person_tracking/results/` in `.txt` format


Run the following code will visualize the tracking results while testing:

`python htc_deepsort.py /your/trackdata/ --display`


 ## Further information
 You can adjust the tracking configuration in `person_tracking/configs/deep_sort.yaml` and detection configuration in  `person_tracking/models/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py`.
