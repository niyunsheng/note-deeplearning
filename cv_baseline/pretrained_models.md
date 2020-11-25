# 预训练模型

[TOC]

## torchvision的预训练模型

官方提供imagenet的[预训练模型](https://pytorch.org/docs/stable/torchvision/models.html)

归一化参数是分别计算三通道的均值和标准差(std)，计算方法为：

```python
import torch
from torchvision import datasets, transforms as T

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
dataset = datasets.ImageNet(".", split="train", transform=transform)

means = []
stds = []
for img in subset(dataset):
    means.append(torch.mean(img))
    stds.append(torch.std(img))

mean = torch.mean(torch.tensor(means))
std = torch.mean(torch.tensor(stds))
```

这个广泛应用的参数是根据imagenet的某个subset计算得出的，而具体是哪些图片已不可查：

```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
```

这里的预训练模型包括，网站提供了这些模型在imagenet上的测试准确度。

* AlexNet
* VGG
* ResNet
* SqueezeNet
* DenseNet
* Inception v3
* GoogLeNet
* ShuffleNet v2
* MobileNet v2
* ResNeXt
* Wide ResNet
* MNASNet

## timm[pytorch-image-models]

[rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)

提供pip包`pip install timm`

```python
import timm
from pprint import pprint
model_names = timm.list_models(pretrained=True)
print(len(model_names))
>>> 263
```

包括263个预训练模型，权重的来源有三种：
* 大多数是有预训练模型的模型，从原始来源获取
* 部分tensorflow的权重，经原始来源转化得到
* 从官网提供的训练脚本和参数训练得到，作者训练的模型结果和参数可以看[这里](https://rwightman.github.io/pytorch-image-models/results/)

值得注意的是，作者给出了各个模型详细的imagenet测试数据[link](https://github.com/rwightman/pytorch-image-models/tree/master/results)

error，我没有发现在哪里可以看到每个模型的具体来源，在库中的[讨论区已提问](https://github.com/rwightman/pytorch-image-models/discussions/287)。

## pretrained-models.pytorch

[Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)

提供了作者复现多种模型，并包含评测的imagenet指标。
模型下载[link](https://github.com/Cadene/pretrained-models.pytorch/tree/master/pretrainedmodels/models)

## fastai

[fast.ai](https://docs.fast.ai/)

Fastai的理念就是：Making neural nets uncool again，让神经网络没那么望而生畏，其课程也是采用项目驱动的方式教学。类似Keras，Fastai不只是将PyTorch功能封装了比较“亲切”的API，而是让PyTorch的强大之处易用了。

暂时只有xresnet50的预训练模型[link](https://github.com/fastai/fastai/blob/a07f271ac6a03cd14ff7f8c031c38527e5b238ed/fastai/vision/models/xresnet.py)

## tensorflow

给出了部分在imagenet上的预训练模型以及模型表现[link](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained)

## keras

给出了部分在imagenet上的预训练模型以及模型表现[link](https://keras.io/api/applications/)

## onnx

[ONNX（Open Neural Network Exchange）](https://github.com/onnx/onnx)是一种针对机器学习所设计的开放式的文件格式，用于存储训练好的模型。它使得不同的人工智能框架（如Pytorch、MXNet）可以采用相同格式存储模型数据并交互。 ONNX的规范及代码主要由微软，亚马逊，Facebook和IBM等公司共同开发。

[预训练的onnx模型](https://github.com/onnx/models)，包含的模型有：
* imagenet的1000类多分类模型
* 目标检测和图像分割
* Body, Face & Gesture Analysis
* Image Manipulation
* Speech & Audio Processing
* Machine Comprehension
* Machine Translation
* Language Modelling
* Visual Question Answering & Dialog
* Other interesting models

## OpenVINO

OpenVINO是intel提出的可以在多种硬件上进行推理加速工具。openVINO提供的[openvinotoolkit/open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/index.md)提供了预训练模型，包括以下类别：
* Object Detection Models
* Object Recognition Models
* Reidentification Models
* Semantic Segmentation Models
* Instance Segmentation Models
* Human Pose Estimation Models
* Image Processing
* Text Detection
* Text Recognition
* Text Spotting
* Action Recognition Models
* Image Retrieval
* Compressed models
* Question Answering
* Machine Translation

## 人脸相关模型

* openVINO提供了人脸检测和reid的预训练模型
* [cmusatyalab/openFace](http://cmusatyalab.github.io/openface/)提供了lua语言的预训练模型
  * 该库年代久远，是2016-01-19发布的，后基本不更新
* [OpenFace 2.2.0](https://github.com/TadasBaltrusaitis/OpenFace)
  * 上述库的更新版本
  * 未提供预训练模型？
* [davidsandberg/facenet](https://github.com/davidsandberg/facenet)
  * 提供了两个inception resnet的预训练模型
  * 最后一次更新是17 Apr 2018


## 其他模型

### EfficientNet-PyTorch

#### official tf 

[tensorflow/tpu](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

网站提供了在imagnet数据集上，不同预处理情况下的预训练模型，模型下载之后，才能看到

#### unofficial pytorch 

[lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)

归一化参数同上，该库提供了imagenet的预训练模型，包括b0-b8，按照论文按照两种方式训练，未提供imagenet上的测试参数。[link](https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py)

另外，该库提供了将官方tf模型转化为pytorch模型的方式。

