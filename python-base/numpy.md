# numpy

[numpy中维度的正确理解姿势](https://blog.csdn.net/lllxxq141592654/article/details/83016497)

[numpy中keepdims的理解](https://blog.csdn.net/lllxxq141592654/article/details/83011056)

* 为什么数学中的向量要用两个[]括起来如`[[1 2 3]]`
* 还有维度为`(3,)`的矩阵
* 重点在于抛弃数学上的长宽的两个维度的概念，这里可以有无穷个维度，维度内的数字也没有什么长宽、水平竖直的概念。
* 这里用np.sum()和np.argmin()进行试验

矩阵的轴(axis)：矩阵的轴与维度对应，维度的第一个元素代表的轴是axis=0，第二个元素代表的轴是axis=1.如维度为(2,3,4)的矩阵axis=0即为2代表的维；再如数学中的2×3矩阵(即维度为(2,3))axis=0即为列（列方向），axis=1即为行（行方向）。

对axis进行指定轴操作会使矩阵降维，使用keepdims=True会使消失的轴保留，并保持维度为1。

array本身并不是一个类，只是为了创建一个ndarray。


`sum(a, axis=None, dtype=None, out=None, keepdims=np._NoValue)`

```
>>> c = np.arange(4)
>>> type(c)
<type 'numpy.ndarray'>
>>> c.shape
(4,)
>>> c[0]
0
>>> type(c[0])
<type 'numpy.int64'>
>>> d = np.random.randint(24,size=[2,3,4])
array([[[ 6, 12, 18,  4],
        [15,  8,  6,  5],
        [ 0, 15, 16,  3]],

       [[15, 16,  2, 11],
        [20,  3, 13, 19],
        [11, 23, 21,  7]]])
>>> d.sum(axis=0).shape
(3, 4)
>>> d.sum(axis=1).shape
(2, 4)
>>> d.sum(axis=2).shape
(2, 3)
>>> d.sum(axis=0,keepdims=True).shape
(1, 3, 4)
>>> d.sum(axis=1,keepdims=True).shape
(2, 1, 4)
>>> d.sum(axis=2,keepdims=True).shape
(2, 3, 1)
>>> d.sum(axis=0,keepdims=True)
array([[[21, 28, 20, 15],
        [35, 11, 19, 24],
        [11, 38, 37, 10]]])
>>> d.sum(axis=1,keepdims=True)
array([[[21, 35, 40, 12]],

       [[46, 42, 36, 37]]])
>>> d.sum(axis=2,keepdims=True)
array([[[40],
        [34],
        [34]],

       [[44],
        [55],
        [62]]])
```

`np.argmin(a,axis=None,out=None)`

找到输入数组a的anis轴上最小值的索引。也可以加深对于axis的理解。

默认输入axis为None，结果是将向量展开为一维向量的坐标。

```
>>> a=np.arange(6).reshape(2,3)
>>> a.argmin()
0
>>> a.argmax()
5
>> a.argmin(axis=0)
array([0,0,0])
>>> a.argmin(axis=1)
array([0,0])
>>> b=np.random.randint(24,size=[2,3,2])
array([[[12, 10],
        [11, 23],
        [ 7, 12]],

       [[ 6, 15],
        [ 3, 20],
        [ 5,  5]]])
>>> b.argmin(axis=0)
array([[1, 0],
       [1, 1],
       [1, 1]])
>>> b.argmin(axis=1)
array([[2, 0],
       [1, 2]])
>>> b.argmin(axis=2)
array([[1, 0, 0],
       [0, 0, 0]])
```

# array的参数

* .ndim:维度
* .shape:各维度的尺寸（2,5）
* .size:元素的个数
* .dtype:元素的类型，比如图片像素的类型是`uint8`

# array数组创建
* np.ones((shape),dtype=np.uint8)
* np.zeros((shape),dtype=np.uint8)
* np.full((shape),val)
* np.eye(n) 单位矩阵
* np.arange([start=0,]stop,[step=1,]dtype=None) 产生数组[start,stop)
* np.linspace(start,stop,num=50,endpoint=True,retstep=False,dtype=None)

创建数组：
* np.empty(shaoe,dtype=float,order='C') dtype表示数据类型，order有C和F两个选项，行优先和列优先，在计算机内存中存储元素的顺序
* np.zeros(shape,dtype,order)
* np.ones(shape,dtype,order)


创建随即数组np.random
* np.random.randn(d0,d1,d2……dn)
d0,d1这些是shape。
返回一个或者一组服从标准正态分布的随机样本值。
* np.random.rand(d0,d1,d2……dn)
使用方法与np.random.randn()函数相同。
通过本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
* numpy.random.randint(low, high=None, size=None, dtype=’l’)
* `random.random()` 返回随机生成的一个实数，它在[0,1)范围内。
输入：
low—–为最小值
high—-为最大值
size—–为数组维度大小
dtype—为数据类型，默认的数据类型是np.int。返回随机整数或整型数组，范围区间为[low,high），包含low，不包含high；
high没有填写时，默认生成随机数的范围是[0，low）

# 数组维度变换
* np.reshape(shape) 需要注意的是reshape并不会保存到原图像
* np.resize(shape)

## 矩阵合并

* `np.concatenate([arr1, arr2],1)`是说水平方向进行拼接，如果不填1就默认是垂直方向拼接

## 矩阵保存

1. 数组以二进制格式保存
np.save和np.load是读写磁盘数组数据的两个主要函数。默认情况下，数组以未压缩的原始二进制格式保存在扩展名为npy的文件中，以数组a为例

np.save("filename.npy",a) 
b = np.load("filename.npy")

# 矩阵打乱

np.random.shuffle(arr)
多维矩阵中，只对第一维（行）做打乱顺序操作

## 数组切片
* arr[-1]
* arr[:3]
* arr[:,0]
* arr[1:3]
* arr[:,:,1]

## 常用函数

* np.exp() 参数是array，不能用math.exp()，这里的参数只能是一维的。
* np.sum()
* np.random.rand(d0,d1,…,dn)产生给定形状的随机数，随机数服从连续均匀分布分布,[0,1)
* numpy.random.randn(d0,d1,…,dn)randn函数返回一个或一组样本，具有标准正态分布。
* np.random.randint(low,high=None,size=None,dtype=‘I’)返回[low,high)之间的随机整数，服从离散均匀分布，如果没有输入high的话，返回[0,low)之间的随机整数。dtype：可取int或int64等


## 矩阵转置`T`

```python
x = np.array([[1,2], [3,4]])
print x    # Prints "[[1 2]
           #          [3 4]]"
print x.T  # Prints "[[1 3]
           #          [2 4]]"

# Note that taking the transpose of a rank 1 array does nothing:
v = np.array([1,2,3])
print v    # Prints "[1 2 3]"
print v.T  # Prints "[1 2 3]"
```

## 点乘和矩阵乘法
和MATLAB不同，* 是元素逐个相乘，而不是矩阵乘法。在Numpy中使用dot来进行矩阵乘法：

```python
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print v.dot(w)
print np.dot(v, w)
# Matrix / vector product; both produce the rank 1 array [29 67]
print x.dot(v)
print np.dot(x, v)
# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print x.dot(y)
print np.dot(x, y)
```

## 广播Boardcasting
广播是一种强有力的机制，它让Numpy可以让不同大小的矩阵在一起进行数学计算。我们常常会有一个小的矩阵和一个大的矩阵，然后我们会需要用小的矩阵对大的矩阵做一些计算。

```python
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print y
```

这样是行得通的，但是当x矩阵非常大，利用循环来计算就会变得很慢很慢。我们可以换一种思路：

```python
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))  # Stack 4 copies of v on top of each other
print vv                 # Prints "[[1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]]"
y = x + vv  # Add x and vv elementwise
print y  # Prints "[[ 2  2  4
         #          [ 5  5  7]
         #          [ 8  8 10]
         #          [11 11 13]]"
```

Numpy广播机制可以让我们不用创建vv，就能直接运算，看看下面例子：

```python
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print y  # Prints "[[ 2  2  4]
         #          [ 5  5  7]
         #          [ 8  8 10]
         #          [11 11 13]]"
```

对两个数组使用广播机制要遵守下列规则：

* 如果数组的秩不同，使用1来将秩较小的数组进行扩展，直到两个数组的尺寸的长度都一样。
* 如果两个数组在某个维度上的长度是一样的，或者其中一个数组在该维度上长度为1，那么我们就说这两个数组在该维度上是相容的。
* 如果两个数组在所有维度上都是相容的，他们就能使用广播。
* 如果两个输入数组的尺寸不同，那么注意其中较大的那个尺寸。因为广播之后，两个数组的尺寸将和那个较大的尺寸一样。
* 在任何一个维度上，如果一个数组的长度为1，另一个数组长度大于1，那么在该维度上，就好像是对第一个数组进行了复制。

```python
# Compute outer product of vectors
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print np.reshape(v, (3, 1)) * w

# Add a vector to each row of a matrix
x = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
print x + v

# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:
# [[ 5  6  7]
#  [ 9 10 11]]
print (x.T + w).T

# Another solution is to reshape w to be a row vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print x + np.reshape(w, (2, 1))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
print x * 2
```
