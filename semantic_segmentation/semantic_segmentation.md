# Semantic Segmentation

| model | conference | paper | first author | institute |
| - | - | - | - | - |
| FCN | CVPR 2015 | Fully Convolutional Networks for
Semantic Segmentation | Jonathan Long/Evan Shelhamer | UC Berkeley |


## FCN

动机和创新点：
* 将CNN用在semantic segmentation中, trained end-to-end, pixelsto-pixels。
* 用全卷积网络代替全连接，可以处理任意的不同size的输入和输出
* 将当前分类网络改编成全卷积网络（AlexNet、 VGGNet和GoogLeNet） 并进行微调设
计了跳跃连接将全局信息和局部信息连接起来， 相互补偿
  * 全局信息解决是什么的问题（分类），局部信息解决在哪里的问题。

实验证明FCN-8s是最佳网络。

模型：跳层连接用的是crop and sum.

上采样：转置卷积。

损失函数用交叉熵，即pixel-wise的多分类，每个pixel都贡献损失。

![](images/fcn-architecture.png)

![](images/fcn-architecture2.png)

## U-Net

模型：增加了很多跨层连接，用的是crop and concat.

上采样：转置卷积。

损失：和FCN一样，给不同pixel加了不同的权重（权重和该像素点离最近的细胞的距离有关，将注意力放在两个细胞的分界位置）。

![](images/unet-architecture.png)

## FusionNet

模型：基本和U-Net一致，但是跨层连接的分辨率保持不变，用的是 sum。
上采样：转置卷积。

![](images/fusionnet-architecture.png)

## DeconvNet

上采样：反卷积+反池化

![](images/deconvnet-deconvolution.png)

![](images/deconvnet-architecture.png)

## SegNet

FCN弊端：忽略了高分辨率的特征图，导致边缘信息的丢失（最好的模型是8s的，没有结合的高分辨率信息）；FCN编码器中有大量参数，但解码器非常的小。

上采样：和fcn不同，利用了pooling中的index信息，不再index的位置补0，之后增加的卷积层也起到了将这些0填充的作用，和转置卷积的差别不是很大，torch.nn有利用upmaxpool函数。如下图所示：

![](images/segnet-upsampleing.png))

提出完全对称的编码器解码器结构

![](images/segnet-architecture.png)

