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


## GoogLeNet V1

v1: Going deeper with convolutions
* 增加网络深度的同时，增加网络宽度，引入Inception模块
* 引入1*1卷积

Inception模块

![](../images/Inception_v1.jpg)

GoogLeNet架构

![](../images/GoogLeNet_v1_architecture.jpg)

![](../images/GoogLeNet_v1_architecture2.jpg)

## GoogLeNet V2

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

## GoogLeNet V3

v3: Rethinking the Inception Architecture for Computer Vision
* 论文中提出3中不同的inception模块，代码中有5种
* 只要一个辅助损失层，在17*17结束之后
* 采用了标签平滑

三种不同的inception结构如下：

![](../images/inception_fig5.png)
![](../images/inception_fig6.png)
![](../images/inception_fig7.png)
![](../images/inception_fig10.png)

Inception V2模型架构如下：

![](../images/inception_v2.png)

Inception V3版本相比V2版本修改的地方有：
1. 采用RMSProp优化方法
1. 采用Label Smoothing正则化方法
2. 采用非对称卷积提取17*17特征图
3. 采用带BN的辅助分类层

标签平滑的公式如下：

![](../images/Label_smoothing.png)

注意，其中交叉熵损失中用p的分布逼近q的分布，公式中的u指均匀分布。

标签平滑的代码在torchvision中没有，不能采用CrossEntropy及其变形，需要重新写这段代码。

```python
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    """
    def __init__(self, eps=0.001):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps

    def forward(self, x, target):
        # CE(q, p) = - sigma(q_i * log(p_i))
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)  # 实现  log(p_i)

        # H(q, p)
        H_pq = -log_probs.gather(dim=-1, index=target.unsqueeze(1))  # 只需要q_i == 1的地方， 此时已经得到CE
        H_pq = H_pq.squeeze(1)

        # H(u, p)
        H_uq = -log_probs.mean()  # 由于u是均匀分布，等价于求均值

        loss = (1 - self.eps) * H_pq + self.eps * H_uq
        return loss.mean()
```

## ResNet

## GoogLeNet V4

## ResNetXt

## DenseNet

## SENet