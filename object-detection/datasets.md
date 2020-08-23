# 物体检测数据集和评价指标

## PASCAL VOC

[pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)

PASCAL VOC为图像分类与物体检测提供了一整套标准的数据集，并从2005年到2012年每年都举行一场图像检测竞赛。 PASCAL全称为Pattern Analysis,Statistical Modelling and Computational Learning， 其中常用的数据集主要有VOC 2007与VOC 2012两个版本。

从voc2007开始，都是有20个类别，类别的组织如下：

![](../images/voc2007_classes.png)

这20个类别中，预测是只输出黑色粗体的类别。

voc2007：train/val/test共有9963张图片，包含24640个已标注的object。2007之前的test是公布的，其后的没有公布。

voc2012：train/val共有11540张图片，包含27450个已被标注的ROI annotated objects。

voc2007和voc2012的图片是互斥的，所以具体用法有：
* 只用VOC2007的trainval 训练，使用VOC2007的test测试
只用VOC2012的trainval 训练，使用VOC2012的test测试，这种用法很少使用，因为大家都会结合VOC2007使用
* 使用 VOC2007 的 train+val 和 VOC2012的 train+val 训练，然后使用 VOC2007的test测试，这个用法是论文中经常看到的 07+12 ，研究者可以自己测试在VOC2007上的结果，因为VOC2007的test是公开的。
* 使用 VOC2007 的 train+val+test 和 VOC2012的 train+val训练，然后使用 VOC2012的test测试，这个用法是论文中经常看到的 07++12 ，这种方法需提交到VOC官方服务器上评估结果，因为VOC2012 test没有公布。
* 先在 MS COCO 的 trainval 上预训练，再使用 VOC2007 的 train+val、 VOC2012的 train+val 微调训练，然后使用 VOC2007的test测试，这个用法是论文中经常看到的 07+12+COCO 。
* 先在 MS COCO 的 trainval 上预训练，再使用 VOC2007 的 train+val+test 、 VOC2012的 train+val 微调训练，然后使用 VOC2012的test测试 ，这个用法是论文中经常看到的 07++12+COCO，这种方法需提交到VOC官方服务器上评估结果，因为VOC2012 test没有公布。

## MS COCO

[Microsoft Common Objects in Context](https://cocodataset.org/#home)

起源于是微软于2014年出资标注的Microsoft COCO数据集，与ImageNet 竞赛一样，被视为是计算机视觉领域最受关注和最权威的比赛之一。

包括三个任务，对应不同的annotations文件。
1. 目标检测和图像分割
2. 图像标注，用一句话描述图片上的信息
3. 人体关键点检测，定位人体在哪里，以及人体的关键点信息

参考：
* [目标检测数据集PASCAL VOC简介](https://arleyzhang.github.io/articles/1dc20586/)