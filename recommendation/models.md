# 推荐模型

推荐模型面临的共性问题有：
* 头部问题：视频之间贫富差距大


推荐算法分类：
* 根据产品的存在形式可以分为：首页推荐、热门推荐和相关推荐等。
* 根据推荐技术的不同分为：基于内容的推荐、基于协同过滤的推荐、基于混合的推荐。
* 根据利用的信息不同可分为：协同过滤推荐、社会化推荐、兴趣点推荐、知识图推荐以及标签推荐等。
* 根据推荐任务不同可分为：评分预测和项目排序。
* 根据模型所利用假设不同分为：以KNN为代表的非训练的方法，以MF为代表的传统机器学习方法，以及以Wide&Deep模型为代表的深度学习推荐等。

<!-- 代码部分来自：[fun-rec](https://github.com/datawhalechina/fun-rec) -->

评分预测在`ml-1m`数据集上进行测试，用mseloss作为评测指标。

## 传统模型

![](images/tradition_models.png)

![](images/models_summary1.png)
![](images/models_summary2.png)

| method[code] | mseloss | logloss |
| - | - | - |
| [UserCF](codes/UserCF.py)|  | |
| [ItemCF](codes/ItemCF.py)|  | |
| MF | | |
| FM | | |
| GBDT+LR | |
| LS-PLM | | |

* CF
    * UserCF
    * ItemCF
    * 可以用稀疏矩阵来表示共现矩阵
    * 相似度的计算有三种方法
        * 余弦相似度
        * 去除user评分均值的相似度
        * 去除物品评分均值的相似度【这种有错误吧！？】
* MF
    * 用`nn.Embedding`来取向量
* FM
    * dense特征取对数，sparse特征进行类别编码
    * 注意稀疏特征相乘时是两个float数字相乘，onehot之后的特征每个维度都对应一个编码向量，比如性别的男和女各对应一个编码向量；但是连续特征只对应一个编码向量
    * FFM引入filed的概念，比如性别的男和女属于一个field，连续特征的年龄属于一个field。FM中的一个编码向量，在这里扩充为filed个。
* GBDT+LR
    * 是特征工程模型化的开端
* LS-PLM，也叫MLR(MixedLogisticRegression，混合逻辑回归)
    * 首先用聚类函数 π对样本进行分类 (这里的 π 采用了 softmax 函数对样本进行多分类)，再用 LR 模型计算样本在分片中具体的 CTR，然后将二者相乘后求和。
    * $f(x)=\sum_{i=1}^{m} \pi_{i}(x) \cdot \eta_{i}(x)=\sum_{i=1}^{m} \frac{\mathrm{e}^{\mu_{i} \cdot x}}{\sum_{j=1}^{m} \mathrm{e}^{\mu_{j} \cdot x}} \cdot \frac{1}{1+\mathrm{e}^{-w_{i} \cdot x}}$
    * 其中的超参数"分片数 " m 可以较好地平衡模型的拟合与推广能力 。 当 m=l 时， LS-PLM 就退化为普通的逻辑回归 。 m 越大，模型的拟合能力越强 。 与此同 时，模型参数规模也随 m 的增大而线性增长，模型收敛所需的训练样本也随之增 长。 在实践中，阿里巴巴给出的m的经验值为12。



## 深度学习模型

![](images/deep_models.png)

![](images/models_summary3.png)
![](images/models_summary4.png)

* AutoRec
    * 可分为基于User和基于Item的自编码器
* DeepCrossing
    * ![](images/DeepCrossing.png)
    * 将所有的Embedding拼接起来，再和dense特征拼接起来
* NeuralCF
    * ![](images/NeuralCF1.png)
    * ![](images/NeuralCF2.png)
    * NeuralCF更像是从MF发展的，是FM增加了另外一路高阶，其实叫NeuralMF更好理解
* PNN
    * PNN虽然也用了DNN来对特征进行交叉组合，但是并不是直接将低阶特征放入DNN中，而是设计了Product层先对低阶特征进行充分的交叉组合之后再送入到DNN中去。
    * 是DeepCrossing的改进，特征之间不直接用拼接的方式
    * ![](images/PNN.png)
    * PNN模型对于深度学习结构的创新主要在于乘积层的引入。具体地说， PNN 模型的乘积层由线性操作部分(图 3-12 中乘积层的 z 部分，对各特征向量进行 线性拼接)和乘积操作部分(图 3-12 中乘积层的 p 部分)组成 。 其中，乘积特 征交叉部分又分为内积操作和外积操作，使用内积操作的 PNN 模型被称为 IPNN (Inner Product-based Neural Network)，使用外积操作的 PNN模型被称为 OPNN ( Outer Product-based Neural Network )。


## Learning to Rank



## DNN双塔