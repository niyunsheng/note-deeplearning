# docker基本概念和操作

[《docker practice》摘要](https://yeasy.gitbook.io/docker_practice/)

Docker 使用 Google 公司推出的 Go 语言 进行开发实现,
基于 Linux 内核的 cgroup，namespace，以及 AUFS 类的 Union FS 等技术，
对进程进行封装隔离，属于操作系统层面的虚拟化技术。

由于隔离的进程独立于宿主和其它的隔离的进程，因此也称其为容器。

传统虚拟机技术是虚拟出一套硬件后，在其上运行一个完整操作系统，在该系统上再运行所需应用进程。

而容器内的应用进程直接运行于宿主的内核，容器内没有自己的内核，而且也没有进行硬件虚拟。

## 安装和配置用户组

ubuntu16.04一键安装脚本

`curl -fsSL get.docker.com -o get-docker.sh`
`sudo sh get-docker.sh --mirror Aliyun`

启动docker

`sudo systemctl enable docker`
`sudo systemctl start docker`

重启和关闭docker

`sudo systemctl restart docker`
`sudo systemctl stop docker`

建立docker用户组并添加用户

`sudo groupadd docker`
`sudo usermod -aG docker $USER`

测试是否安装成功

`docker run hello-world`

配置国内镜像加速

`sudo vim /etc/docker/daemon.json`（不存在则创建）
写入以下内容：

```json
{
    "registry-mirrors": [
        "https://registry.docker-cn.com"
    ]
}
```

之后重新启动服务：
`sudo systemctl daemon-reload`
`sudo systemctl restart docker`

## 基本概念

### 镜像

docker镜像`image`是一个特殊的文件系统，除了提供容器运行时所需的程序、库、资源、配置等文 件外，还包含了一些为运行时准备的一些配置参数（如匿名卷、环境变量、用户等）。镜像 不包含任何动态数据，其内容在构建之后也不会被改变。

镜像构建时，会一层层构建，前一层是后一层的基础。每一层构建完就不会再发生改变，后 一层上的任何改变只发生在自己这一层。比如，删除前一层文件的操作，实际不是真的删除 前一层的文件，而是仅在当前层标记为该文件已删除

### 容器

镜像`image`和容器`container`的关系，就像是面向对象程序设计中的 类 和 实例一样，
**镜像是静态的定义，容器是镜像运行时的实体。**容器可以被创建、启动、停止、删除、暂停等。

容器的实质是进程，但与直接在宿主执行的进程不同，容器进程运行于属于自己的独立的命名空间。

每一个容器运行时，是以镜像为基础层，在其上创建一个当前容器的存储层。

容器存储层的生存周期和容器一样，容器消亡时，容器存储层也随之消亡。因此，任何保存于容器存储层的信息都会随容器删除而丢失。

按照 Docker 最佳实践的要求，容器不应该向其存储层内写入任何数据，容器存储层要保持无 状态化。
所有的文件写入操作，都应该使用 数据卷`Volume`、或者绑定宿主目录，
在这些 位置的读写会跳过容器存储层，直接对宿主（或网络存储）发生读写，其性能和稳定性更高。

数据卷的生存周期独立于容器，容器消亡，数据卷不会消亡。

### 仓库`repository`

镜像构建完成后，可以很容易的在当前宿主机上运行，但是，如果需要在其它服务器上使用 这个镜像，我们就需要一个集中的存储、分发镜像的服务，Docker Registry 就是这样的服务。

我们可以通过 `<仓库名>:<标签>` 的格式来指定具体是这个软件哪个版本的镜像。如果不给 出标签，将以 latest 作为默认标签。

每个仓库可以包含多个标签（ Tag ）；每个标签对应一个镜像。

## 基本镜像操作

* 获取镜像

`docker pull [选项] [Docker Registry 地址[:端口号]/]仓库名[:标签]`

如`docker pull ubuntu:16.04`，默认从docker hub下载镜像。

* 列出镜像

`docker image ls`
列表包含了仓库名、标签 、镜像 ID、创建时间以及所占用的空间。

注意：docker hub显示的大小是压缩之后的大小，而上述命令显示的大小是本地的大小。

* 查看镜像、容器、数据卷所占用的空间

`docker system df`

* 删除虚悬镜像

虚悬镜像是已经失去了存在的价值，是可以随意删除的
`docker image prune`

* 根据条件列出镜像

根据仓库名列出镜像`docker image ls ubuntu`
其他列出部分镜像的规则请查看书籍。

* 删除镜像

`docker image rm [选项] <镜像1> [<镜像2> ...]`

<镜像>可以是镜像短ID(一般3个字符以上即可区别于其他镜像)、镜像长 ID、镜像名或者镜像摘要。

删除行为有两种，包括`untagged`和`deleted`，当我们使用上面命令删除镜像的时候，实际上是在要求删除某个标签的镜像，因为一个镜像可以对应多个标签，因此当我们删除了所指定的标签后，可能还有别的标签指 向了这个镜像，如果是这种情况，那么 Delete 行为就不会发生。

用`docker image ls`配合删除操作，比如，我们需要删除所有仓库名为redis的镜像`docker image rm $(docker image ls -q redis)`

## 基本容器操作

* 查看运行的容器信息

`docker container ls`

* 查看包括终止状态的容器信息

`docker container ls -a`

* 终止运行中的容器

`docker container stop`

终止所有运行中的容器`docker container stop $(docker container ls -q)`

* 启动容器

启动容器输出hello world并终止容器
`docker run ubuntu:16.04 /bin/echo 'Hello world'`

启动容器进入bash终端，允许用户交互
`docker run -t -i ubuntu:16.04 /bin/bash`

-t选项让docker分配一个伪终端并绑定到容器的标准输入上，-i则让容器的标准输入保持打开。

**docker容器在运行完成命令之后，容器即变为终止状态**。

* 后台运行

启动容器并后台运行：使用`-d`参数

如果输入以下命令，则会在当前命令行执行程序并输出。

`docker run ubuntu:16.04 /bin/sh -c "while true; do echo hello world; sleep 1; done"`

加参数`-d`之后可以后台运行。

`docker run -d ubuntu:16.04 /bin/sh -c "while true; do echo hello world; sleep 1; done"`

获取运行中容器的输出信息，用`docker container logs`命令。

* 启动已终止的容器

`docker container start`

* 重启运行中的容器

`docker container restart`

* **进入后台运行的容器**

可以使用`attach`或者`exec`命令

如果用`attach`进入容器，在stdin中使用exit，会导致容器的停止。

使用`exec`命令可以开启命令行。
`docker exec -it id bash`
后可跟多个参数，如果只用`-i`参数，由于没有分配伪终端，界面没有我们熟悉的 Linux 命令提示符，但命令执行结果仍然可以返回。
当 `-i -t` 参数一起使用时，则可以看到我们熟悉的 Linux 命令提示符。

从`exec`命令进入容器，然后exit，并不会导致容器的停止。

* 删除处于终止状态的容器

`dockre container rm`

* 清楚所有处于终止状态的容器

`docker container prune`

## 数据管理

docker中管理数据有两种方式：数据卷volumes和挂载主机目录bind mounts.

### 数据卷

数据卷是一个可供一个或多个容器使用的特殊目录，它绕过 UFS，可以提供很多有用的特性：

* 可以在容器之间共享和重用
* 对 数据卷 的修改会立马生效
* 对 数据卷 的更新，不会影响镜像
* 数据卷 默认会一直存在，即使容器被删除

* 创建数据卷

`docker volumes create my-vol`

* 查看所有的数据卷

`docker volume ls`

* 查看数据卷信息

`docker volume inspect my-vol`

* 启动容器时可以挂载多个数据卷

在`docker run`命令时，可以用`--mount`标记来将数据卷挂载到容器里

`docker run -d -P --name ubuntu-vol --mount source=my-vol,target=/webapp ubuntu:16.04 python app.py`

* 删除数据卷

`docker volume rm my-vol`

数据卷 是被设计用来持久化数据的，它的生命周期独立于容器，Docker 不会在容器被删除后 自动删除 数据卷 ，并且也不存在垃圾回收这样的机制来处理没有任何容器引用的 数据卷。

如果需要在删除容器的同时移除数据卷。可以在删除容器的时候使用 `docker rm -v `这个命令。

* 删除无主的数据卷

`docker volume prune`

## 挂载主机目录

* 挂载主机目录作为数据卷

使用`--mount`标记可以指定挂载一个本地主机的目录到容器中去。

`docker run -d -P --name ubuntu-vol --mount type=bind,source=/src/ubuntu-vol,target=/webapp ubuntu:16.04 python app.py`

加载主机上的`/src/ubuntu-vol`目录到容器的`/webapp`目录。注意，本地目录必须是绝对路径。如果本地目录不存在，docker会报错。

默认的挂载权限是读写，可以增加readonly指定为只读。

`docker run -d -P --name ubuntu-vol --mount type=bind,source=/src/ubuntu-vol,target=/webapp,readonly ubuntu:16.04 python app.py`

* 挂载一个本地主机文件作为数据卷

```bash
docker run --rm -it \
# -v $HOME/.bash_history:/root/.bash_history \
--mount type=bind,source=$HOME/.bash_history,target=/root/.bash_history \
ubuntu:16.04 \
bash
```

这样就可以记录在容器输入过的命令了。

## Compose

Compose项目是 Docker 官方的开源项目，负责实现对 Docker 容器集群的快速编排。Compose 定位是 「定义和运行多个 Docker 容器的应用（Defining and running multicontainer Docker applications）」，其前身是开源项目 Fig。

Compose允许用户通过一个单独的 docker-compose.yml 模板文件（YAML 格式）来定义一组相关联的应用容器为一个项目（project）。

两个重要的概念：
* 服务 ( service )：一个应用的容器，实际上可以包括若干运行相同镜像的容器实例。
* 项目 ( project )：由一组关联的应用容器组成的一个完整业务单元，在 `docker-compose.yml`文件中定义。

Compose 的默认管理对象是项目，通过子命令对项目中的一组容器进行便捷地生命周期管理。

* 运行compose项目

首先需要编写`docker-compose.yml`文件，然后在该文件目录中运行`docker-compose up`，该命令十分强大，它将尝试自动完成包括构建镜像，（重新）创建服务，启动服务，并关联 服务相关容器的一系列操作。

大部分时候都可以直接通过该命令来启动一个项目。

默认情况， `docker-compose up` 启动的容器都在前台，控制台将会同时打印所有容器的输出信息，可以很方便进行调试。

当通过 Ctrl-C 停止命令时，所有容器将会停止。

如果使用 `docker-compose up -d` ，将会在后台启动并运行所有的容器。一般推荐生产环境下使用该选项。