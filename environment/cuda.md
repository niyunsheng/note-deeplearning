# cuda架构

CUDA是一个并行计算平台和编程模型，能够使得使用GPU进行通用计算变得简单和优雅。
在CUDA架构之下，开发人员可以通过`CUDA C`对GPU进行编程，`CUDA C`是对标准C的简单扩展。

早期与GPU交互的唯一接口是标准图形接口，如OpenGL和DirectX，因此要在GPU上执行运算，就受限制于图形API的编程模型。研究人员研究如何通过图形API执行通用计算，这也使得他们的计算问题在GPU看来仍然是传统的渲染问题。

nvidia Driver：nvidia提供了一些软件来实现应用程序与支持cuda的硬件之间的通信。安装了设别驱动，就能够运行编译好的`CUDA C`代码。

CUDA toolkit：由于CUDA C应用程序将在两个不同的处理器上执行计算，因此需要两个编译器。其中nvidia提供了GPU编译器，CPU编译器在linux上用的是GUN gcc。

cuda程序有两种代码，一种是运行在cpu上的host代码，一种是运行在gpu上的device代码

## 安装显卡驱动

* 查看GPU型号`lspci | grep -i nvidia`

直接在官网选择对应的操作系统和显卡型号下载安装即可。

* 安装完成后可以查看驱动版本`cat /proc/driver/nvidia/version`

## CUDA toolkit

CUDA Toolkit由以下组件组成：

* Compiler: CUDA-C和CUDA-C++编译器NVCC位于bin/目录中。它建立在NVVM优化器之上，而NVVM优化器本身构建在LLVM编译器基础结构之上。因此开发人员可以使用nvm/目录下的Compiler SDK来直接针对NVVM进行开发。
* Tools: 提供一些像profiler,debuggers等工具，这些工具可以从bin/目录中获取
* Libraries: 下面列出的部分科学库和实用程序库可以在lib/目录中使用(Windows上的DLL位于bin/中)，它们的接口在include/目录中可获取。
  * cudart: CUDA Runtime
  * cudadevrt: CUDA device runtime
  * cupti: CUDA profiling tools interface
  * nvml: NVIDIA management library
  * nvrtc: CUDA runtime compilation
  * cublas: BLAS (Basic Linear Algebra Subprograms，基础线性代数程序集)
  * cublas_device: BLAS kernel interface
  * ...
* CUDA Samples: 演示如何使用各种CUDA和library API的代码示例。可在Linux和Mac上的samples/目录中获得

安装toolkit之前要安装Driver: 运行CUDA应用程序需要系统至少有一个具有CUDA功能的GPU和与CUDA工具包兼容的驱动程序。**每个版本的CUDA工具包都对应一个最低版本的CUDA Driver**，也就是说如果你安装的CUDA Driver版本比官方推荐的还低，那么很可能会无法正常运行。

特别辨析：
* nvcc：nvcc其实就是CUDA的编译器,可以从CUDA Toolkit的/bin目录中获取,类似于gcc就是c语言的编译器。
* nvidia-smi：它是一个基于前面介绍过的NVIDIA Management Library(NVML)构建的命令行实用工具，旨在帮助管理和监控NVIDIA GPU设备。

## CUDA toolkit 安装

**注意**：如果使用conda，则conda自带cudatoolkit和cudnn，无序单独安装cudatoolkit和cudnn。

参考官方安装指南[cuda-installation-guide-linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

1. 安装前确认是否有支持cuda的GPU、GCC等。
2. 禁用nouveau驱动。
3. 重启系统`ctrl+alt+F1`进入命令行模式。
4. 确认未加载nouveau驱动。
5. 在线deb包安装
   1. 注意:运行`sudo apt-get install cuda-9.0`命令时，如果后面不加版本号，则默认安装最新版本。

安装完成后用`cat /usr/local/cuda/version.txt`查看cuda版本号，用`nvcc -V`查看nvcc的版本号，用`nvidia-smi`查看GPU使用情况（如果该命令不可用，应该是环境变量没配置的问题）。

基本概念解析：
* nouveau驱动：nouveau是LINUX内核中NVIDIA显卡的开源驱动，但是它不支持CUDA，所以安装cuda驱动前需禁用该驱动
* OpenGL：OpenGL是Khronos Group开发维护的一个规范，它主要为我们定义了用来操作图形和图片的一系列函数的API，需要注意的是OpenGL本身并非API。GPU的硬件开发商则需要提供满足OpenGL规范的实现，这些实现通常被称为“驱动”，它们负责将OpenGL定义的API命令翻译为GPU指令。

## 安装cuDNN

cuDNN其实就是一个专门为深度学习计算设计的软件库，里面提供了很多专门的计算函数，如卷积等。

[官方文档](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/#installlinux-tar)

[登录下载](https://developer.nvidia.com/rdp/cudnn-archive)对应的cuDNN版本。

查看cudnn版本号`cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2`

## `nvidia-smi`命令

提供监控GPU使用情况和更改GPU状态的功能，是一个跨平台工具，它支持所有标准的NVIDIA驱动程序支持的Linux发行版以及从WindowsServer 2008 R2开始的64位的系统。

`nvidia-smi`命令显示所有GPU的当前状态信息：
* fan：风扇转速
* temp：GPU温度，GPU温度过高会导致GPU频率下降
* perf：性能状态，从P0(最大性能)到P12(最小性能)
* Pwr：GPU功耗
* Persistence-M：持续模式的状态（持续模式耗能大，但在新的GPU应用启动时花费时间更少）
* Bus-Id：GPU总线，domain:bus:device.function
* Disp.A：Display Active，表示GPU的显示是否初始化
* Memory-Usage：显存使用率
* Volatile GPU-Util：GPU使用率
* ECC：是否开启错误检查和纠正技术，0/DISABLED, 1/ENABLED
* Compute M.：计算模式，0/DEFAULT,1/EXCLUSIVE_PROCESS,2/PROHIBITED

* `nvidia-smi -l xxx`动态刷新，不写xxx时默认5s刷新一次
* `nvidia-smi -f xxx`将查询的信息输出到具体的文本中，不在终端显示
* `nvidia-smi -q`查询所有GPU的当前详细信息


## `gpustat`命令

`pip install gpustat`

可以用`gpustat -cpui`实时监视运行情况