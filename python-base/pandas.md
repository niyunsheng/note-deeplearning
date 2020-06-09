# pandas

* [Pandas官方-Pandas 10分钟入门](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html)
* [pandas索引和选择数据](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html)

## Pandas一维数据结构：Series

Series 是一维带标记的数组结构，可以存储任意类型的数据（整数，浮点数，字符串，Python 对象等等）。

作为一维结构，它的索引叫做 index，基本调用方法为

`s = pd.Series(data, index=index)`
其中，data 可以是以下结构：字典、ndarray、标量（例如 5）。
index 是一维坐标轴的索引列表。

如果 data 是个 ndarray，那么 index 的长度必须跟 data 一致：
`s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])`

`pd.Series(np.random.randn(5))`

简单的向量操作 series 与 ndarray 的表现一致,区别在于Series 的操作默认是使用 index 的值进行对齐的，而不是相对位置，对于两个不能完全对齐的 Series，结果的 index 是两者 index 的并集，同时不能对齐的部分当作缺失值处理。

## Pandas二维数据结构：DataFrame

```python
def write_list_to_csv(csv_path,index_list):
    test=pd.DataFrame(columns=None,data=index_list)
    test.to_csv(csv_path,mode='a+', header=False,index=False)

def read_csv_to_list(csv_path):
    return pd.read_csv(csv_path,header = None).values.tolist()
```

读取csv文件：`df=pd.read_csv('test.csv',header='infer',sep=',')`