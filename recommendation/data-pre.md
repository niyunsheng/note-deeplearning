# 数据预处理

可以参考`tf.feature_column`中的处理方法。

![](./images/tf-columns.jpeg)

* numeric_column 数值列，最常用。
* bucketized_column 该函数将连续变量进行分桶离散化，输出one-hot的结果，方便连续值指标与分类变量进行交叉特征构建
* categorical_column_with_identity 分类标识列，one-hot编码，相当于分桶列每个桶为1个整数的情况。
* categorical_column_with_vocabulary_list 分类词汇列，one-hot编码，由list指定词典。
* categorical_column_with_vocabulary_file 分类词汇列，由文件file指定词典。
* categorical_column_with_hash_bucket 哈希列，整数或词典较大时采用。
* indicator_column 指标列，由Categorical Column生成，one-hot编码
* embedding_column 嵌入列，由Categorical Column生成，嵌入矢量分布参数需要学习。嵌入矢量维数建议取类别数量的 4 次方根。
* crossed_column 交叉列，可以由除categorical_column_with_hash_bucket的任意分类列构成。


