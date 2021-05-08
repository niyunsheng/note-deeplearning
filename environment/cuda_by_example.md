# CUDA实战

参考
* [CUDA官方入门教程](https://devblogs.nvidia.com/even-easier-introduction-cuda/)

* [其他书籍笔记：GPU高性能编程 CUDA实战](https://blog.csdn.net/fishseeker/article/details/75093166)

## 总览

[如何学习cuda/知乎小小将](https://zhuanlan.zhihu.com/p/34587739)

GPU并不是一个独立运行的计算平台，而需要与CPU协同工作，可以看成是CPU的协处理器，因此当我们在说GPU并行计算时，其实是指的基于CPU+GPU的异构计算架构。

CUDA编程模型是一个异构模型，需要CPU和GPU协同工作。在CUDA中，host和device是两个重要的概念，我们用host指代CPU及其内存，而用device指代GPU及其内存。CUDA程序中既包含host程序，又包含device程序，它们分别在CPU和GPU上运行。同时，host与device之间可以进行通信，这样它们之间可以进行数据拷贝。典型的CUDA程序的执行流程如下：

1. 分配host内存，并进行数据初始化；
2. 分配device内存，并从host将数据拷贝到device上；
3. 调用CUDA的核函数在device上完成指定的运算；
4. 将device上的运算结果拷贝到host上；
5. 释放device和host上分配的内存。

cuda中通过函数类型限定词区别开host和device上的函数，主要的三个函数类型限定词如下：

* `__global__`：在device上执行，从host中调用（一些特定的GPU也可以从device上调用），返回类型必须是void，不支持可变参数参数，不能成为类成员函数。注意用`__global__`定义的kernel是异步的，这意味着**host不会等待kernel执行完就执行下一步**。`__global__`标识符高速编译器，函数应该编译为在设备而不是主机上运行。
* `__device__`：在device上执行，单仅可以从device中调用，不可以和`__global__`同时用。
* `__host__`：在host上执行，仅可以从host上调用，一般省略不写，不可以和`__global__`同时用，但可和`__device__`同时用，此时函数会在device和host都编译。

编译CUDA程序`nvcc test_gpu.cu -o test_gpu`

执行CUDA程序`./test_gpu`

运行程序前添加命令 `CUDA_VISIBLE_DEVICES=0` ，可以使得只有0号设备可用。也可以用命令 `CUDA_VISIBLE_DEVICES=0,2,3`

## 3 CUDA C简介

目标：
* 了解为host编写的代码和为device编写的代码的区别
* 如何用主机上运行设备代码
* 如何在支持cuda的设备上使用设备内存
* 如何查询系统中支持cuda的设备的信息

### device函数调用

CUDA C的优势在于，它提供了与C语言在语言级别上的集成，因此这个设备函数调用看上去非常像主机函数调用。

CUDA C的简单性及其强大功能在很大程度上都来源于它淡化了主机代码和设备代码之间的差异。

### 传递参数

* 可以像调用C函数那样将参数传递为设备函数
* 当设备上执行任何有用的操作时，都需要分配内存，例如将计算值返回给主机。

分配内存`cudaMalloc()`，第一个参数是一个指针，指向用来保存新分配的内存地址的变量，第二个参数是分配内存的大小。

**主机代码可以将cuda上的指针作为参数传递，但是，不能使用这个指针来读取或者写入内存。**

访问设备内存，需要用到内存复制`cudaMemcpy()`，其中，最后一个参数`cudaMemcpyDeviceToHost`，表示源指针是一个设备指针，目标指针是一个主机指针。`cudaMemcpyHostToDevice`含义相反。

### 查询设备

查询GPU设备属性。

```c++
int count;
cudaGetDeviceCount(&count);

cudaDeviceProp prop;
for(int i=0;i < count;i++){
    cudaGetDeviceProperties(&prop,i);
    printf("%s",prop.name);
    // 等设备属性
}
```

## 4 CUDA C并行编程

目标：
* 了解CUDA在实现并行性时采用的一种重要方式
* 用CUDA C编写一段并行代码。

用CUDA编写代码是很容易的，但是GPU的应用前景很大程度上取决于能否从许多问题上发掘出大规模并行性。

一个gpu函数`kernel<<<blocks,threads>>>()`，第一个参数表示线程块（block）的个数，第二个参数表示每个线程块中创建的线程数量。程序如何知道正在运行的是哪一个线程块呢？就需要调用预定义变量`blockIdx.x`，cuda支持二维的线程块数组，对于二维空间的计算问题，例如矩阵数学计算或者图像处理，使用二维索引带来更多便利。同理，用`threadIdx.x`来表示线程块内不同的线程。

```c++
__global__ void add( int *a, int *b, int *c ) {
    int tid = blockIdx.x;    // this thread handles the data at its thread id
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}
```

这个程序中，不同block中的程序，通过预定义变量对数组的不同元素执行操作。

`blocks`的数量最大是65535，这是硬件的限制。
`threads`的数量最大是512，即每个线程块内线程的数量最大为512。

`gridDim`是一个常数，表示block的大小，`gridDim.x`表示blocks的x维度的大小。

## 5 线程协作

目标：
* 学习CUDA C中的线程
* 了解不同线程之间的通信机制
* 了解并执行线程的同步机制

### 线程块和线程

`kernel<<<blocks,threads>>>()`第一个参数表示线程块的个数，第二个参数表示每个线程块内部线程的个数。

如果只用1个线程块，里面开N个线程（N<=512），即`add<<<1,N>>>( dev_a, dev_b, dev_c );`可以对最大为512维的向量求和。其中，向量的下标为：
`int tid = threadIdx.x;`

如果需要对更大的向量，比如128*128维求和，就需要用更多的线程块，即`add<<<128,128>>>( dev_a, dev_b, dev_c );`，其中，求和部分的代码如下：

```c++
__global__ void add( int *a, int *b, int *c ) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}
```

对于任意长度的向量，可以这样调用`add<<<(N+127)/128,128>>>(dev_a, dev_b, dev_c );`。

但是，由于线程块的数量也有限65535，所以，最大求值的向量长度为65535*512，如果超过了这个长度，上述方案就会失效。

让一个线程块的每个线程不止进行一次运算，而是进行多次运算。

正如多CPU或者多核版本中，每次递增的数量不是1，而是CPU的数量，现在，GPU当中，每次递增的数目是所有线程的数目。

```c++
__global__ void add( int *a, int *b, int *c ) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}
```

### 共享内存与同步

在编写代码时，可以将CUDA C的关键字`__share__`添加到变量声明中，这将使这个变量驻留在共享内存中。

对于GPU上启动的每个线程块，CUDA C编译器都将创建该变量的一个副本，线程块中的每个线程共享这块内存，但线程却无法看到也不能修改其他线程块的变量副本。线程块共享内存使得一个线程块的多个线程能够在计算上进行通信和协作。

对线程块中的线程进行同步，用到的操作是`__syncthreads();`，这个函数确保线程块中的每个线程都执行完`__syncthreads();`前面的语句后，才会执行下一条语句。

规约操作：对256个数字求和，只需要8次循环即可。

一个线程块内，由于分支条件的存在，有些线程需要执行一条指令，其他线程不需要执行时，这种情况叫做线程发散`Thread Divergence`。在正常的环境中，发散的线程只会使得某些线程处于空闲状态，而其他线程将执行分支中的代码。

不能把`__syncthreads();`写到分支语句当中，因为CUDA架构将确保，除非线程块中的每个线程都执行了`__syncthreads();`语句，否则没有任何线程能够执行`__syncthreads();`之后的语句。

```c++
__global__ void dot( float *a, float *b, float *c ) {
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float   temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    // set the cache values
    cache[cacheIndex] = temp;
    
    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}
```

## 6 常量内存与事件

目标：
* 了解如何在CUDA C中使用常量内存。
* 了解常量内存的性能特性。
* 学习如何使用CUDA事件来测量应用程序的性能。

前面已经看到了CUDA C程序中使用全局内存和共享内存，其中，全局内存用`cudaMalloc()`分配空间，在各个device函数中，用指针进行读写；共享内存是线程块之间的共享区域，用`__shared__`标识符。

常量内存指的是在核函数执行期间不会发生变化的数据，NVIDIA提供了64KB的常量内存，并且对常量内存采取了不同于标准全局内存的处理方式。在某种情况下，用常量内存来替代全局内存能够有效的减少内存带宽。

常量内存的声明方法和共享内存类似，在变量前面加`__constant__`修饰符，而且将其写在函数的外部作为全局变量。例如：
`__constant__ Sphere s[20];`

常量内存的写法不需要对该数组调用`cudaMalloc()`或者`cudaFree()`，而是在编译时为这个数组提交一个固定的大小。

当把主机的数据复制到device的常量内存当中，需要用到`cudaMemcpyToSymbol()`。

从常量内存读取数据与全局内存相比，性能提升的主要原因有两个：
* 对常量内存的单词操作可以广播到其他的“临近（Nearby）”线程，这将节约约15次读取操作。
* 常量内存的数据将缓存起来，因此对相同地址的连续读操作将不会产生额外的内存通信量。

在CUDA架构中，线程束是指一个包含32个线程的集合，这个线程集合被“编织在一起”并且以“步调一致（Lockstep）”的形式执行。在程序中的每一行，线程束中的每个线程都将在不同的数据上执行相同的指令。

在处理常量内存时，NVIDIA硬件将单词内存读取操作广播到每个半线程束（half-warp）。在半个线程束中包含16个线程。如果在半个线程束中的每个线程都从常量内存的相同地址中读取大量的数据，那么GPU只会产生一次读取请求并在随后将数据广播到每个线程。如果从内存常量中读取大量的数据，那么这种方式产生的内存流量只是使用全局内存时间的1/16.

另外，因为是常量内存，所以硬件将主动将这个常量数据缓存在GPU中，在第一次从常量内存的某个地址读取之后，当其他半线程束请求同一个地址时，将命中缓存，这同样减少了额外的内存流量。

但是，如果16个线程读取不同的地址时，实际上会降低性能。因为只有读取相同地址，才值得将该读取操作广播到16个线程。然而，如果半线程束中的所有16个线程需要访问常量内存中不同的数据，那么这个16次不同的操作会被串行化，从而需要16倍的时间发出请求。但如果从全局内存读取，那么这些请求会同时发出。在这种情况下，从常量内存读取就慢于从全局内存中读取。

测试cuda程序性能的流程如下:

```c++
    // capture the start time
    cudaEvent_t     start, stop;
    cudaEventCreate( &start ) );
    cudaEventCreate( &stop ) );
    cudaEventRecord( start, 0 ) );
    /****
    要执行的程序
    ****/
    // get stop time, and display the timing results
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float   elapsedTime;
    cudaEventElapsedTime( &elapsedTime,start, stop );
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );

    cudaEventDestroy( start );
    cudaEventDestroy( stop );
```

PS：以下工作暂不更新

## 7 纹理内存

## 8 图形互操作性

## 9 原子性

## 10 流

## 11 多GPU系统上的CUDA C
