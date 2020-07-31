# Faiss

* [facebookresearch/faiss](https://github.com/facebookresearch/faiss)
* [faiss C++ API](https://rawgit.com/facebookresearch/faiss/master/docs/html/annotated.html)
* [官网介绍翻译](https://www.infoq.cn/article/2017/11/Faiss-Facebook)

## 简介

Faiss是facebook于15年开发，17年3月发布的一个相似性搜索类库。是第一个基于十亿高维向量构建的`k-nearest-neighbor graph`。

faiss由C++编写，并且提供了和numpy完美衔接的python接口，对一些核心算法提供了GPU实现。

Faiss唯一的硬依赖是`BLAS/Lapack`。它是使用Intel MKL开发的。

## 相似性搜索

np提供的函数`np.argmin(a,axis=None,out=None)`可以找到输入数组a的anis轴上最小值的索引。

给定维度d中的一组向量x_i，Faiss从中构建数据结构。在构造结构之后，当给定维度为d的新向量x时，它有效地执行操作：

$$i = argmin_i || x - x_i ||$$

这里用的是L2距离（平方求和再开方）

* Faiss可以返回K个最近邻
* Faiss一次搜索几个向量而不是一个（批处理）。对于许多索引类型，这笔搜索一个接一个的矢量更快。
* 执行最大内积搜索(`maximum inner product search`)而不是最小欧几里得搜索(`minimum Euclidean search`)
* 返回查询点的给定半径范围内的所有元素

## 安装

[install.md](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md)

### C++编译安装

1. `./configure`命令，生成`makefile.inc`文件
  *  `./configure --without-cuda`CPU版本
  *  `./configure --with-cuda=/path/to/cuda` 加入cuda路径。
  * 一般情况下cuda路径为`/usr/local/cuda/`
  * `./configure` 是用来检测你的安装平台的目标特征的。比如它会检测你是不是有CC或GCC，并不是需要CC或GCC，它是个shell脚本。
2. `make`命令，生成静态链接库`.a`和动态链接库`.so`
  * 如果gpu版本报错信息`nvcc fatal : unsupported gpu architecture 'compute 75'`。需要编辑上一步生成的`makefile.inc`文件，注释掉`-gencode=arch=compute_75,code=compute_75`这一行
  * make 是用来编译的，它从Makefile中读取指令，然后编译
  * `make -j8`在多核cpu上，并行编译是能够提高编译速度的
  * make是编译报错`can't find -lopenblas`，这时候先去`/usr/lib/`下查找libopenblas这个库，可以看到有`libopenblasp-r0.2.18.so`，但是没有`libopenblas.so`，设置一个软链接即可`sudo ln -s libopenblasp-r0.2.18.so libopenblas.so`
3. `make install`
* `make install`是用来安装的，它也从Makefile中读取指令，安装到指定的位置。
4. `make py`
* 遇到错误信息如下：编译cpu版本：configure: error: An implementation of BLAS is required but none was found.
* 这个错误的原因是缺少BLAS库，安装这个库即可。

```
git clone https://github.com/xianyi/OpenBLAS
cd OpenBLAS
sudo make FC=gfortran //如果没有安装gfortran，需要用sudo apt-get install gfortran
sudo make PREFIX=/usr/ install 表示安装到系统文件夹
```

这样就解决了编译faiss的问题。

> 缺少库文件的问题，要么是没安装，要么是没加入makefile的路径里面。

> 设置软链接：`ln `


### anaconda+python安装

最简单的方式使用conda安装，在Linux和OSX上都支持faiss-cpu。在Linux系统上提供了使用`CUDA8 / CUDA9 / CUDA10`编译的`faiss-gpu`

我安装是GPU的CUDA10版本

`conda install faiss-gpu cudatoolkit=10.0 -c pytorch # For CUDA9`

注意：截止2020-07-30，`faiss-gpu`不支持cuda10.1，所以，需要用conda新创建一个安装cuda10.0的环境


## Faiss Index 基础

* [官方get started](https://github.com/facebookresearch/faiss/wiki/Getting-started)

生成向量

```python
import numpy as np
d = 64 # dimension
nb = 100000 # database size
nq = 10000  # nb of queries
np.random.seed(1234) # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.
```

### L2暴力搜索`IndexFlatL2`

```python
import faiss                   # make faiss available
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)

k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
```

### 加速:分段+倒排索引`IndexIVFFlat`

nlist是设置聚类中心的个数，nprobe是查询时设置查询的聚类的个数，当nprobe=nlist，可以获取和暴力搜索同样的结果。

```python
nlist = 100
k = 4
quantizer = faiss.IndexFlatL2(d)  # the other index
index = faiss.IndexIVFFlat(quantizer, d, nlist)
assert not index.is_trained
index.train(xb)
assert index.is_trained

index.add(xb)                  # add may be a bit slower as well
D, I = index.search(xq, k)     # actual search
print(I[-5:])                  # neighbors of the 5 last queries
index.nprobe = 10              # default nprobe is 1, try a few more
D, I = index.search(xq, k)
print(I[-5:])                  # neighbors of the 5 last queries
```

### 减少内存占用：用PQ方法`IndexIVFPQ`

向量仍然存储在`nlist`个倒排文件中，但是用乘积量化将每个向量的代销压缩为m(d是m的倍数)

```python
nlist = 100
m = 8                             # number of subquantizers
k = 4
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                                    # 8 specifies that each sub-vector is encoded as 8 bits
index.train(xb)
index.add(xb)
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
index.nprobe = 10              # make comparable with experiment above
D, I = index.search(xq, k)     # search
print(I[-5:])
```

### 在GPU上运行

* 单GPU

```python
res = faiss.StandardGpuResources()  # use a single GPU
# build a flat (CPU) index
index_flat = faiss.IndexFlatL2(d)
# make it into a gpu index
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

gpu_index_flat.add(xb)         # add vectors to the index
print(gpu_index_flat.ntotal)

k = 4                          # we want to see 4 nearest neighbors
D, I = gpu_index_flat.search(xq, k)  # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
```

* 多GPU

```python
ngpus = faiss.get_num_gpus()

print("number of GPUs:", ngpus)

cpu_index = faiss.IndexFlatL2(d)

gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
    cpu_index
)

gpu_index.add(xb)              # add vectors to the index
print(gpu_index.ntotal)
k = 4                          # we want to see 4 nearest neighbors
D, I = gpu_index.search(xq, k) # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
```

## Faiss Index 进阶

重点：
* IVF倒排文件对文件聚类，划分聚类中心，然后生成倒排索引文件，提高检索速度
* IVF包含粗量化器，将聚类中心加入到粗量化器中
* PQ用于压缩向量，但是也造成了查找的不准确

[Faiss-indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)

| Method | Class name | index_factory | Main parameters | Bytes/vector | Exhaustive | Comments     |
| - | - | - | - | - | - | - |
| Exact Search for L2  | IndexFlatL2 | "Flat"   | d    | 4*d     | yes        | brute-force  |
| Exact Search for Inner Product | IndexFlatIP | "Flat"   | d    | 4*d     | yes        | also for cosine (normalize vectors beforehand)    |
| Hierarchical Navigable Small World graph exploration     | IndexHNSWFlat | 'HNSWx,Flat`  | d, M | 4*d + 8 * M  | no | |
| Inverted file with exact post-verification  | IndexIVFFlat  | "IVFx,Flat" | quantizer, d, nlists, metric | 4*d     | no | Take another index to assign vectors to inverted lists |
| Locality-Sensitive Hashing (binary flat index) | IndexLSH   | -        | d, nbits        | nbits/8 | yes        | optimized by using random rotation instead of random projections  |
| Scalar quantizer (SQ) in flat mode  | IndexScalarQuantizer    | "SQ8"    | d    | d       | yes        | 4 bit per component is also implemented, but the impact on accuracy may be inacceptable |
| Product quantizer (PQ) in flat mode | IndexPQ    | "PQx"    | d, M, nbits     | M (if nbits=8)       | yes        | |
| IVF and scalar quantizer       | IndexIVFScalarQuantizer | "IVFx,SQ4" "IVFx,SQ8" | quantizer, d, nlists, qtype  | SQfp16: 2 * d, SQ8: d or SQ4: d/2 | no | there are 2 encodings: 4 bit per dimension and 8 bit per dimension        |
| IVFADC (coarse quantizer+PQ on residuals)   | IndexIVFPQ | "IVFx,PQy"  | quantizer, d, nlists, M, nbits  | M+4 or M+8 | no | the memory cost depends on the data type used to represent ids (int or long), currently supports only nbits <= 8 |
| IVFADC+R (same as IVFADC with re-ranking based on codes) | IndexIVFPQR | "IVFx,PQy+z"  | quantizer, d, nlists, M, nbits, M_refine, nbits_refine | M+M_refine+4 or M+M_refine+8      | no | |

### Faiss MetricType

常用的有`METRIC_INNER_PRODUCT`和`METRIC_L2`，前者可以用于计算cosine距离，不过前提是在添加向量之间对向量进行L2归一化。

### Flat Indexes

Flat类型的index不对向量进行压缩，将他们存储在`ntotal * code_size`的数组中，在搜索时，所有向量都被顺序解码，并且与查询向量进行比较，比较的结果插入到堆中，最后返回最近邻结果。

所以，Flat类型的Index不压缩特征向量，搜索是穷尽的。

需要注意的是
* Flat类型的index不支持`add_with_ids`，只支持按照顺序添加向量。
* 但是，可以用`IndexIDMap`封装以添加该功能，代码如下。
* 支持删除`remove`，但是删除后会缩小索引并且改变编号。


* c++ 

```cpp
faiss::IndexIDMap<faiss::IndexFlatL2> index(dim);
```

* python

```python
import faiss
index = faiss.IndexFlatL2(feature_dims)
index = faiss.IndexIDMap(index)
```

### 压缩向量

* 不编码：`Flat`:不压缩向量就存储
* 16位浮点数编码：`IndexScalarQuantizer`，其中的元素`QuantizerType`为`QT_fp16`，可能导致精度损失
* 8/6/4整数编码：`IndexScalarQuantizer`，其中的元素`QuantizerType`为`QT_8bit/QT_6bit/QT_4bit`，量化为256/64/16级的整数
* PQ编码：`IndexPQ`，向量被量化为子向量，每个子向量被量化为几个位（通常为8位）

### 单元探针（cell-probe）方法(`IndexIVF*`)

采用k-means之类的聚类方法加速搜索，代价是不能准确的找到最近邻。

* 特征空间被划分为`nlist`个聚类/单元，聚类算法训练`nlist`个聚类中心，然后将所有特征加入到`nlist`个倒排文件中
* 这些聚类中心存储在粗量化器(也是一个索引)中
* 数据库中的向量被分配给这些单元之一，并存储在`nlist`反向列表组成的反向列表文件结构中
* 在查询时，选择`nprobe`个倒排列表，如果`nprobe=nlist`，则获取和暴力相同的结果
* 将查询与分配给这些列表的每个数据库向量进行比较

构造函数中包含一个粗量化器`quantizer`，用于对反向列表进行分配

### `Flat`属性的index作为粗量化器的`cell-probe`方法

通常，将`Flat`索引用作粗量化器，将`IndexIVF*`中的聚类算法训练的聚类中心添加到`Flat`索引当中，`nprobe`在查找时设定，默认为1，表示要查找的聚类的个数，即只在这些单元中进行查找。

单元的数目，可以按照这个经验表达`nlist = c * sqrt(n)` ，c是一个常数，表示列表的不均匀分布，可以设定为10。

### 带有乘积量化的index

* C++

```cpp
#include <faiss/IndexPQ.h>
#include <faiss/IndexIVFPQ.h>

// Define a product quantizer for vectors of dimensionality d=128,
// with 8 bits per subquantizer and M=16 distinct subquantizer
size_t d = 128;
int M = 16;
int nbits = 8;
faiss:IndexPQ * index_pq = new faiss::IndexPQ (d, M, nbits);

// Define an index using both PQ and an inverted file with nlists to avoid exhaustive search
// The index 'quantizer' must be already declared
faiss::IndexIVFPQ * ivfpq = new faiss::IndexIVFPQ (quantizer, d, nlists, M, nbits);

// Same but with another level of refinement
faiss::IndexIVFPQR * ivfpqr = new faiss::IndexIVFPQR (quantizer, d, nclust, M, nbits, M_refine, nbits);
```

* python

```python
m = 16 # number of subquantizers
n_bits = 8     # bits allocated per subquantizer
pq = faiss.IndexPQ (d, m, n_bits)        # Create the index
pq.train (x_train)  # Training
pq.add (x_base) # Populate the index
D, I = pq.search (x_query, k)  # Perform a search

# Inverted file with PQ refinement
coarse_quantizer = faiss.IndexFlatL2 (d)
index = faiss.IndexIVFPQ (coarse_quantizer, d,
 ncentroids, code_size, 8)
index.nprobe = 5
```
