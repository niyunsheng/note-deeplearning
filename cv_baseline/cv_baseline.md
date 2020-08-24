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

v1: Going deeper with convolutions
* 增加网络深度的同时，增加网络宽度，引入Inception模块
* 引入1*1卷积

Inception模块

![](../images/Inception_v1.jpg)

GoogLeNet架构

![](../images/GoogLeNet_v1_architecture.jpg)

![](../images/GoogLeNet_v1_architecture2.jpg)

v2: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
* 提出BN层用于解决数据偏移的问题（内部协变量偏移），加速网络收敛
* 开启标准化层用于神经网络的时代
  * 在Batch Normalization基础上拓展出了一系列标准化网络层，如Layer Normalization（LN），Instance Normalization（IN），Group Normalization（GN）
* 相对v1的改变：
  * 卷积层和激活层之间增加BN层
  * 将 5\*5 卷积更改为两个 3\*3 卷积

BatchNorm计算过程

![](../images/batchnorm.png)

反向传播计算公式

![](../images/batchnorm_backpropagation.png)

BN训练过程

![](../images/batchnorm_training.png)

注意：
* bn层中每个channel一个均值一个方差，channel的均值和方差的调整过程遵循动量的原则。
* 可查看代码 [BatchNorm_test](./BatchNorm_test.py)

`mean_new = momentum * mean_batch + (1 - momentum) * mean_old`

`var_new = momentum * var_batch + (1 - momentum) * var_old`

* 推理阶段，采用训练阶段的最后一个batch之后的mean和var，推理阶段的mean和var不改变。

BN推荐参考资料[莫烦ptyhon:什么是批标准化](https://mofanpy.com/tutorials/machine-learning/tensorflow/intro-batch-normalization/)

