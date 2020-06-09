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

## 安装CUDA toolkit

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