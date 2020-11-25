# pytorch 1.7.0a

[官方文档](https://pytorch.org/docs/stable/index.html)

[中文学习资料pytorch-handbook](https://github.com/zergtant/pytorch-handbook)

核心开发者Edward Z. Yang的一次原理讲解[PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/)

A Tour of PyTorch Internals (Trevor Killeen)
* [part1](https://pytorch.org/blog/a-tour-of-pytorch-internals-1/)
* [part2](https://pytorch.org/blog/a-tour-of-pytorch-internals-2/)

## Tensor

![tensor_illustration](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/tensor_illustration.png)

## 动态图：自动求导

![Dynamic graph](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/dynamic_graph.gif)

## 代码库结构

[codebase-structure](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md#codebase-structure)

重要的四个文件夹
* `c10` 源自`caffe 10`，核心库文件，该库仅包含基本功能，包括Tensor和Storage数据结构的实际实现
* `aten` `A Tensor Library`这是pytorch的C++张量库(不支持autograd)
  * `src`
    * `TH THC THUCNN` 原版torch代码，这里的代码逐渐会移动到 ATen 中的native
    * `ATen` **这里是最重要的**
      * `core` ATen的核心功能，这里的代码会逐步迁移到c10文件夹
      * `native` **运算符的现代实现**，编写新的运算符应在此进行
* `torch` 实际的pytorch库，所有不在csrc中的内容都是一个python模块
  * `csrc` 组成pytorch库的C++文件，该目录树中的文件混合了python绑定代码和C++的工作，`setup.py`中有绑定文件的规则列表，这里的文件通常以 python_ 开头
  * `_c`模块 

