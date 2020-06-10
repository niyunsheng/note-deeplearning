# anaconda

## 安装

* 在tuna上下载anaconda[安装包](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive)

* 在终端执行命令安装`bash Anaconda-name.sh`

* 换源

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/
conda config --set show_channel_urls yes
```

* 新建环境

新建环境
`conda create --name ML python=3.7`

好处在于不用更新conda，这个环境出错的话，可以把这个环境全部删除了，而base环境无法删除



## 基本操作

* 更新conda `conda update conda`
* 更新所有第三方包 `conda update --all`
* 查看conda信息`conda info`
* 进入base环境`conda activate base`
* 创建新环境 `conda create --name ML python=3.7`
* 注意，这样安装的环境不会安装python，输入`which python`命令后会显示系统的Python，`/usr/bin/python`，但是，安装完`conda install numpy`,再输入`which python`后就显示了。
* 查看所有的环境`conda info --envs`
* 激活新环境`conda activate ML`
* 如果conda没有添加到PATH，则用`source anaconda3/bin/activate`来激活环境
* 退出base环境`conda deactivate`
* 查看包`conda list`
* 安装包`conda instll numpy`
* 卸载包`conda uninstall numpy`
* 安装tensorflow`pip install tensorflow`
* 安装keras`pip install keras`
* pip升级包`pip install --upgrade numpy`
* 删除环境`conda remove -n name --all`
* 卸载anaconda：直接删除anaconda3文件夹即可。

## 配置jupyter notebook和jupyter lab

1. 一般情况下anaconda中自带jupyter notebook，直接在终端使用`jupyter notebook`即可进入
2. 二者配置相同，在终端`jupyter lab`进入
3. 自己新创建的环境中一般没有配置，用`conda install jupyter`安装即可
4. 配置远程服务
    * 生成配置文件`jupyter notebook --generate-config`
    * 修改配置文件`vim ~/.jupyter/jupyter_notebook_config.py`，在文件头部添加如下字段：

```python
c.NotebookApp.allow_remote_access = True
c.NotebookApp.notebook_dir = 'd:\\' #更改默认打开根目录
c.NotebookApp.ip='*' #意思是任意IP都可以访问
c.NotebookApp.open_browser = False  #意思是默认不打开浏览器
c.NotebookApp.port =8888 #随便指定一个你想要的端口,后面可以从这个端口使用
```

在云主机上配置jupyter lab 服务，一般需要在配置文件中添加
`c.NotebookApp.allow_remote_access = True`

* 后台运行jupyter lab

`nohup jupyter notebook > /dev/null 2>&1 &`

* jupyter notebook导出pdf

最好的方法是导出为html或者markdown，然后再转换为pdf。

## conda和pip

Conda是一种通用包管理系统，旨在构建和管理任何语言的任何类型的软件。因此，它也适用于Python包。
Conda和pip服务于不同的目的，并且只在一小部分任务中直接竞争：即在孤立的环境中安装Python包。

在conda环境中一般会有pip，用`which pip`命令可以查看到当前用的pip即为当前conda环境中的pip。

* pip换源

临时修改：

`pip install package -i https://pypi.tuna.tsingahua.edu.cn/simple`

* 永久修改：

修改` ~/.pip/pip.conf `(没有就创建一个)，内容如下：

```
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```