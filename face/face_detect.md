# 人脸检测

## MTCNN：Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks

pdf:https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf

突出点：
* multi-task：
  * detection（面部/非面部分类、边界框回归）
  * alignment（关键点定位）
* 级连网络
  * a fast Proposal Network (P-Net):通过浅CNN快速生成候选窗口
  * a Refinement Network (R-Net):通过更复杂的CNN拒绝大量非面部窗口来细化窗口
  * The Output Network (O-Net):使用更强大的CNN再次细化结果并输出五个面部标志位置
* 特征金字塔
  * 按照resize_factor（0.7左右即可）对test图片进行resize，直到大等于Pnet要求的12*12大小。这样子你会得到原图、原图*resize_factor、原图*resize_factor^2...、原图*resize_factor^n（注，最后一个的图片大小会大等于12）这些不同大小的图片。
  * 注意，这些图像都是要一幅幅输入到Pnet中去得到候选的。

![](./images/MTCNN-arc.png)
![](./images/MTCNN-arc2.png)

训练的损失函数
* 面部/非面部分类
  * $L_{i}^{\text {det }}=-\left(y_{i}^{\text {det }} \log \left(p_{i}\right)+\left(1-y_{i}^{\text {det }}\right)\left(1-\log \left(p_{i}\right)\right)\right)$
* 边界框回归
  * $L_{i}^{b o x}=\left\|\hat{y}_{i}^{b o x}-y_{i}^{b o x}\right\|_{2}^{2}$
* 关键点回归
  * $L_{i}^{\text {landmark }}=\left\|\hat{y}_{i}^{\text {landmark }}-y_{i}^{\text {landmark }}\right\|_{2}^{2}$
* 合并的损失
  * $\min \sum_{i=1}^{N} \sum_{j \in\{\text { det }, \text { box,landmark }\}} \alpha_{j} \beta_{i}^{j} L_{i}^{j}$
  * $\alpha$表示三个任务的损失权重分布，$\beta$表示正负样本，P-Net和R-Net中使用$\alpha_{det}=1,\alpha_{box}=0.5,\alpha_{landmark}=0.5$，O-Net中使用$\alpha_{det}=1,\alpha_{box}=0.5,\alpha_{landmark}=1$


训练过程：
* 原图resize成图片金字塔
* 每张图片输入P-Net输出`m*n*16`的输出，对所有预测为人脸的图片的排序在一起（这样首先删除了预测不为人脸的框），进行nms，即首先用分类的预测分数排序，然后删除box的iou大于0.6的预测值较小的框。这样剩下的框在原图中裁好，按照长边裁正方形，然后resize为24*24.
* 将P-Net得到的所有图每个都分别输入，同样的过程映射到原图然后进行nms，再减少一部分候选框，同样是长边裁正方形，然后resize为48*48.

* 三个网络分别训练，每个网络都有自己想要的大小的图片即训练集，Pnet是`12*12*3`， Rnet是`24*24*3`， Onet是`48*48*3`的图片。然后每个网络都是单独训练的，即训练好Pnet后训练Rnet，再训练Onet，前者的训练输出是后者的训练输入。


对于正负样本在训练和推理阶段
正样本：三个网络均关注三个损失
负样本：即背景，只计算面部/非面部分类损失，另外两个损失设置为0。

三个网络中，对于三种损失的权重是不一样的，



开源pyhton库`facenet_pytorch`中测试的fps如下：





mtcnn算法只有一个缺点，就是当图像中的人脸数目比较多的时候，mtcnn人脸检测算法的的性能下降的比较快，而retinaFace算法不受人脸数量的限制，这是由于 mtcnn算法使用了图像金字塔算法，需要对图像进行多次缩放，导致前向运算次数比较多，严重拖慢了检测速度，而retinaFace是基于FPN的检测算法，无需进行图像金字塔，仅需要前向运算一次即可，好了说了那么多，下面跟着小编的步伐，从应用的角度比较一下。



## RetinaFace

retinaFace，也是目前one-stage 目标检测算法的缺点

1.预设的anchor 是一柄双刃剑，anchor需要事先指定，并且不同的检测任务需要的anchor并不一样

2.anchor的数量会非常的多，由于fpn根据featureMap上的每一个点生成一个anchor,即使我们用nms或者soft-nms去重，但是负样本的数量依然非常非常多，与正样本比例严重失衡，所以诸如RetinaNet等网络的工作都是想办法去采取合适的比例参数平衡这个差异。

3.Anchor数量巨多，需要每一个都进行IOU计算，耗费巨大的算力，降低了效率，步骤十分繁琐，而这些冗余其实是可以消灭的。以上是anchor非常明显的缺点，所以anchor-free模型开始兴起，它的兴起，一方面说明anchor-based固有的缺陷需要改正；另一方面说明现有的anchor-based方法已经有了很高的baseline，不好继续突破。



原文網址：https://kknews.cc/tech/bzopvnm.html