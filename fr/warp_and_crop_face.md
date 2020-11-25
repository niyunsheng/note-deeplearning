# 二d人脸对齐

注意：这里说的人脸对齐和传统的Face Alignment差别较大，传统的Face Alignment是指从人脸图片上检测人脸关键点，然后用放射变换将人脸归正。而本文说的人脸对齐只是指仿射变换的部分。

> 代码文件[`align_faces.py(foamliu)`](https://github.com/foamliu/InsightFace-PyTorch/blob/master/align_faces.py)

> 如不加以说明，下文中出现的代码均来源于该代码文件

人脸对齐可分为如下步骤：
1. 由目标人脸大小获得关键点的目标位置
2. 由检测到的人脸关键点和目标关键点位置得到仿射变换的变换矩阵
3. 进行仿射变换

## 获取目标关键点位置

当人脸大小为`(96/w，112/h)`时，目标关键点的位置为

```python
REFERENCE_FACIAL_POINTS = [
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 92.3655014],
    [62.72990036, 92.20410156]
]
DEFAULT_CROP_SIZE = (96, 112)
```

函数接口为`get_reference_facial_points(output_size=None,inner_padding_factor=0.0,outer_padding=(0, 0),default_square=False) -> ndarray`

有四个参数：
* `default_square`
  * 程序最先处理这个参数，如果为true，则`DEFAULT_CROP_SIZE`会修改为`(112,112)`，五个关键点坐标也会相应的向左平移`(112-96)//2=8`个像素
* `output_size`
* `inner_padding_factor`
* `outer_padding`

整体变化的过程中包括平移和缩放，内padding和外padding的数值作为参数给出，缩放的系数由这些参数推算得到。**先内padding，然后缩放，然后外padding。**

后面三个参数之间是有关联的，不恰当的输入程序会报错，三者之间的约束可以表示为：

`(DEFAULT_CROP_SIZE * (1 + inner_padding_factor * 2))` 与 `output_size - outer_padding` 等比例，这个比例可以用`scale_factor`表示，故下式成立：

```
(DEFAULT_CROP_SIZE * (1 + inner_padding_factor * 2)) * scale_factor + outer_padding
= output_size
```

关键点的位置由`REFERENCE_FACIAL_POINTS`变为：

```python
size_diff = tmp_crop_size * inner_padding_factor * 2
tmp_5pts += size_diff / 2

tmp_crop_size += np.round(size_diff).astype(np.int32)
scale_factor = (output_size[0] - outer_padding[0])/tmp_crop_size[0]
tmp_5pts = tmp_5pts * scale_factor

tmp_5pts += np.matrix(outer_padding)
```

## 得到变换矩阵

得到变换矩阵共有三种方法：
* 方法1：最小二乘法
* 方法2：由三点对应获得仿射变换矩阵
* 方法3：相似性变换

### 最小二乘法

```python
def get_affine_transform_matrix(src_pts, dst_pts):
    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])

    A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_)

    if rank == 3:
        tfm = np.float32([
            [A[0, 0], A[1, 0], A[2, 0]],
            [A[0, 1], A[1, 1], A[2, 1]]
        ])
    elif rank == 2:
        tfm = np.float32([
            [A[0, 0], A[1, 0], 0],
            [A[0, 1], A[1, 1], 0]
        ])

    return tfm
```

函数的输入`src_pts`和`dst_pts`均为`ndarray:(5,2)`
输出变换矩阵为`ndarray:(3,2)`

`np.linalg`是numpy中的线性代数库，
`np.linalg.lstsq(a, b, rcond='warn')`lstsq 是 LeaST SQuare （最小二乘）的意思，即求使得`|| ax - b ||`最小的x的值。函数返回一个包含四个值的元组：
* 第一元素表示所求的最小二乘解x
  * 使得 $a.dot(x) - b$ 的L2模最小
* 第二个元素表示残差总和
* 第三个元素表示x矩阵的秩
* 第四个元素表示x的奇异值

### 通过三个点即可获得仿射变换矩阵

`tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])`



### 相似性变换矩阵

```python
tform = trans.SimilarityTransform()
tform.estimate(src_pts, ref_pts)
tfm = tform.params[0:2, :]
```

相似变换的矩阵如下：

$$\left[\begin{matrix}s * cos\theta & s * {(-sin\theta)} & t_{x} \\ s * sin\theta & cos\theta & t_{y} \\ 0 & 0 & 1\end{matrix}\right]$$

相比仿射变换，相似变换的自由度较小，左上角2×2矩阵为旋转部分，tx和ty为平移因子，它有4个自由度，即旋转，x方向平移，y方向平移和缩放因子s。

## 补充知识：仿射变换

> 本部分内容参考《计算机图形学导论》第五章：几何变换

三维空间的变换矩阵是一个`3*3`的矩阵`W`, $P2 = W.dot(P1)$ , P1的坐标可以写作 $(x,y,1)$，其中第三维表示bias。

### 平移

将 $(x,y)$ 平移到 $(x+t_x,y+t_y)$ , 变换矩阵为：

$$
\left[\begin{matrix} x & y & 1 \end{matrix}\right] * \left[\begin{matrix}1 & 0 & 0 \\ 0 & 1 & 0 \\ t_x & t_y & 1\end{matrix}\right] = \left[\begin{matrix} x + t_x & y + t_y & 1 \end{matrix}\right]
$$

### 缩放

将 $(x,y)$ 缩放到 $(x*s_x,y*s_y)$ , 变换矩阵为：

$$
\left[\begin{matrix} x & y & 1 \end{matrix}\right] * \left[\begin{matrix}s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1\end{matrix}\right]= \left[\begin{matrix} x * s_x & y * s_y & 1 \end{matrix}\right]
$$

### 旋转

将 $(x,y)$ 顺时针旋转 $\theta$ 角度

$$
\left[\begin{matrix} x & y & 1 \end{matrix}\right] * \left[\begin{matrix}cos{\theta} & sin{\theta} & 0 \\ -sin{\theta} & cos{\theta} & 0 \\ 0 & 0 & 1\end{matrix}\right] = \left[\begin{matrix} x * cos{\theta} - y * sin{\theta} & x * sin{\theta} + y * cos{\theta} & 1 \end{matrix}\right]
$$

### 错切

$$
\left[\begin{matrix} x & y & 1 \end{matrix}\right] * \left[\begin{matrix}1 & a & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1\end{matrix}\right] = \left[\begin{matrix} x & a*x+y & 1 \end{matrix}\right]
$$

### 复合变换

仿射变换公有6个自由度，两个旋转因子，两个缩放因子，x和y的平移因子，共6个。

以 $(t_x,t_y)$ 为中心旋转 $\theta$ 度，先平移 $(-x,-y)$，然后以原点为中心旋转，最后再平移 $(x,y)$

$$
\left[\begin{matrix} x & y & 1 \end{matrix}\right] * \left[\begin{matrix}1 & 0 & 0 \\ 0 & 1 & 0 \\ -t_x & -t_y & 1\end{matrix}\right] * \left[\begin{matrix}cos{\theta} & sin{\theta} & 0 \\ -sin{\theta} & cos{\theta} & 0 \\ 0 & 0 & 1\end{matrix}\right] * \left[\begin{matrix}1 & 0 & 0 \\ 0 & 1 & 0 \\ t_x & t_y & 1\end{matrix}\right]
$$

### opencv中的仿射变换

`crop_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))`

其中`tfm`是`shape=(2,3)`的变换矩阵

