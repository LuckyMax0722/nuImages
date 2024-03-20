
## Introduction
In this project, we propose an approach for data preprocessing based on nuImages database.

## Installation
1. Environment requirements

* Ubuntu 20.04
* Python 3.8
* Pytorch 2.1.2
* CUDA 12.1

The following installation guild suppose ``Ubuntu=20.04`` ``python=3.8`` ``pytorch=2.1.2`` and ``cuda=12.1``. You may change them according to your system, but linux is mandatory.

1. Create a conda virtual environment and activate it.
```
conda create -n nuimages python=3.8
conda activate nuimages
```

2. Clone the repository.
```
git clone https://github.com/LuckyMax0722/nuImages.git
```

3. Install the pyTorch
```
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

4. Install the nuImages development kit
```
pip install nuscenes-devkit
```

For detail information, please refer to [nuImages
/nuscenes_devkit](https://github.com/LuckyMax0722/nuImages/blob/51132df94d060667b071b24f462db95cc29c0294/nuscenes_devkit/README.md)

5. Install MM development kit
```
pip install -U openmim
mim install mmengine
mim install mmcv
```
6. Install the dependencies.
```
pip install easydict
pip install notebook
pip install tensorboard
pip install pandas
pip install seaborn thop
```

7. yolo2coco
```
pip install pylabel
```
## Data Preparation

## Tasks
### One-Stage Multi-Object Detection
#### YOLOv5

```
cd ./nuImages/YOLOv5
python3 train.py --img-size 640 --batch-size 6 --epochs 100 --data /home/jiachen/nuImages/YOLOv5/data/nuImages.yaml --cfg /home/jiachen/nuImages/YOLOv5/models/yolov5x_nuImages.yaml --weights weights/yolov5x.pt
```

#### YOLOv7
```
cd ./nuImages/YOLOv7
python train.py --workers 8 --device 0 --epochs 30 --batch-size 6 --data ./data/nuImages.yaml --img 640 640 --cfg ./cfg/training/yolov7x_nuImages.yaml --weights ./weights/yolov7x.pt --name yolov7x --hyp ./data/hyp.scratch.p5.yaml
```