# CV Baseline

## AlexNet

* 数据：ImageNet
* 算法：GPU
* 模型：深度卷积神经网络

![](../images/alexnet.jpg)


## VGG

* 全部是`kernel=3*3，padding=1`的小卷积核，卷积操作不改变feature map的分辨率
* 加深网络
* 可以将网络表示为`[[Conv2d+ReLU]*m+MaxPool2d]*n + Linear`
* 如果增加bn层，将`[Conv2d+ReLU]`修改为`[Conv2d+BatchNorm2d+ReLU]`

![](../images/vgg.jpg)


## GoogLeNet

v1:
* 增加网络深度的同时，增加网络宽度，引入Inception模块
* 引入1*1卷积

Inception模块

![](../images/Inception_v1.jpg)

GoogLeNet架构

![](../images/GoogLeNet_v1_architecture.jpg)

![](../images/GoogLeNet_v1_architecture2.jpg)