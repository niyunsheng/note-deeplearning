# 装机指南ubuntu

# 制作u盘启动盘

下载`Ubuntu16.04.06-desktop-amd64.iso`
* [tuna开源镜像ubuntu-releases](https://mirrors.tuna.tsinghua.edu.cn/ubuntu-releases/)
* [网易开源镜像站ubuntu-releases](http://mirrors.163.com/ubuntu-releases/)


* 启动模式分为legacy和UEFI，如果启动盘系统无法识别，可能是只识别UEFI模式。
* 用老毛桃等PE制作的启动盘 **不能安装ubuntu系统**，可以用来制作修复启动选项
* 用UltraISO软碟通制作ubuntu启动盘
    * 打开ISO文件
    * 启动->写入硬盘镜像
    * 格式化U盘，文件系统改为NTFS
    * 写入方式选择USB-HDD+
    * 不用设置便捷启动
    * 写入：看到刻录成功提示即可！

# 分区和安装

较为合理的方式是分一个主分区、一个交换分区和几个逻辑分区，如下表所示。

[参考文章](https://blog.csdn.net/u012052268/article/details/77145427/)

|目录|建议大小|格式|描述|
|-   |-      |-   |-  |
|/   |200G   |ext4|根目录|
|swap|物理内存两倍|swap|交换分区|
|/boot|2G|ext4|内核以及引导系统所需要的文件|
|/tmp|5G左右|ext4|临时文件|
|/home|剩下的全部|ext4|用户工作目录；个人配置文件，如个人环境变量等；所有账号分配一个工作目录|

用df -h 查看

# 配置用户

1. 设置root密码

刚安装好root用户是没有密码的，使用`sudo passwd`，这是让输入密码，即为设置root密码

2. 创建可以登录图形用户界面的用户

`sudo adduser nys`
然后根据系统提示进行密码和注释性描述的配置，全程不用自己输入其他命令即可配置成功，用户主目录和命令解析程序都是系统自动指定。

3. 修改用户密码

`passwd usrname`

4. 将该用户添加到sudo用户组
   * 方法1：修改 `/etc/sudoers` 文件；
   * 方法2：`usermod -a -G sudo nys//注意改成你自己的用户名`
   采用方法2，查看`/etc/group`文件，可以看到sudo行多了nys用户，`sudo:x:27:user1,user2,nys`，删除nys用户即可去除sudo权限

5. 删除用户

`sudo userdel -r nys`包括删除相应文件夹

# 配置ssh服务

1. 配置服务

   * 查看本机ip地址`ifconfig`。设置外网访问服务器，需要在路由器配置访问端口映射到内网的服务器的ssh端口。
   * `sudo apt-get install openssh-server`
   * 查看ssh服务是否启动：`sudo ps -e |grep ssh`
   * 如果有sshd,说明ssh服务已经启动
   * 如果没有启动，输入`sudo service ssh start`
   * 如果想通过ssh登录root账户，需要修改配置文件`/etc/ssh/sshd_config`,用“#”号注释掉"PermitRootLogin without-password"，增加一句"PermitRootLogin yes"

2. 设置ssh登录快捷方式

修改`~/.ssh/config`文件，修改如下：

```
Host test
    HostName 192.168.12.123
    User niyunsheng
    IdentityFile /Users/niyunsheng/.ssh/id_rsa
```

3. 设置免密登录

修改`vim .ssh/authorized_keys`，将需要远程登录的主机公钥复制进来

# 开启工作区

右键->更改桌面背景->行为->开启工作区

# 软件仓库、linux换源及更新

每个LINUX的发行版，比如UBUNTU，都会维护一个自己的软件仓库，我们常用的几乎所有软件都在这里面。这里面的软件绝对安全，而且绝对的能正常安装。

为了更快的更新速度，需要先换源
* 备份原来的源`sudo cp /etc/apt/sources.list /etc/apt/sources_init.list`
* 更换源，将阿里源复制进去，**并且删除原来的源就好了**`sudo vim /etc/apt/sources.list`

```
deb http://mirrors.aliyun.com/ubuntu/ xenial main
deb-src http://mirrors.aliyun.com/ubuntu/ xenial main

deb http://mirrors.aliyun.com/ubuntu/ xenial-updates main
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-updates main

deb http://mirrors.aliyun.com/ubuntu/ xenial universe
deb-src http://mirrors.aliyun.com/ubuntu/ xenial universe
deb http://mirrors.aliyun.com/ubuntu/ xenial-updates universe
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-updates universe

deb http://mirrors.aliyun.com/ubuntu/ xenial-security main
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-security main
deb http://mirrors.aliyun.com/ubuntu/ xenial-security universe
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-security universe
```

或者用清华的源

```
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
```

* 查看已安装软件

除此之外，也可以用`rpm`或`dpkg`软件包管理器来查看，它的功能类似于Windows里面的“添加/删除程序”。`rpm -qa`列出所有被安装的rpm包。`dpkg -l`列出所有被安装的dpkg包。Ubuntu采用的是dpkg软件安装方式。

* 更新软件包列表`apt-get update`.这个命令，会访问源列表里的每个网址，并读取软件列表，然后保存在本地电脑。我们在新立得软件包管理器里看到的软件列表，都是通过update命令更新的。
* 以上命令之后，就可以安装vim`apt-get install vim`
* 更新软件包`apt-get upgrade`.这个命令，会把本地已安装的软件，与刚下载的软件列表里对应软件进行对比，如果发现已安装的软件版本太低，就会提示你更新。

* 安装卸载软件

```
apt-get install xxx 安装xxx
apt-get remove xxx 卸载xxx
apt-get remove -purge xxx 卸载xxx同时删除配置文件

对于`.deb`安装包，用dpkg工具进行安装
dpkg -i | --install xxx.deb 安装deb安装包
dpkg -r | --remove xxx.deb 删除安装包
dpkg -r -p |--purge xxx.deb 连同配置文件一起删除
dpkg -I | -info xxx.deb 产看软件包信息
dpkg -L xxx.deb 查看文件拷贝信息
dpkg -l 查看系统中以安装软件包信息
dpkg-reconfigure xxx 重新配置软件包
```

# 配置静态IP

* `ifconfig`查看当前IP和网卡信息。当前可看到网卡名`enp1s0f0`的IP地址为192.168.1.188
* 修改`vim /etc/network/interfaces`

```
auto enp1s0f0
iface enp1s0f0 inet static
address 192.168.1.188
netmask 255.255.255.0
gateway 192.168.1.1 
# 这里一定要设置为.1，设置为.2就错了
```

* 配置dns服务器（配置阿里的）`vim /etc/resolvconf/resolv.conf.d/base`

```
nameserver 223.5.5.5
nameserver 223.6.6.6
```

刷新配置`resolvconf -u`

完成后重新启动网络即可.
`sudo /etc/init.d/networking restart`


# 永远不要用`rm`命令

## 在linux上用`trash`命令
* 安装：`sudo apt-get install trash-cli`
* 删除文件或者文件夹`trash file/dir`
* 查看回收站`trash-list`
* 清空回收站`trash-empty`
* 恢复文件，首先用`restore-trash`命令，然后输入文件序号
* linux上的回收站位置`~/.local/share/Trash`，删除的文件在`files`文件夹下
* 禁用`rm`
  * 修改文件`vim ~/.bashrc`
  * 添加`alias rm="echo 'use trash instead,or the full path: /bin/rm'"`

## 在mac上用`rmtrash`
* 安装：`brew install rmtrash`
* 删除`rmtrash file/dir`
* 查看或者恢复文件会回收站工具
* 禁用`rm`
  * 修改文件`vim ~/.bashrc`
  * 添加`alias rm="echo 'use rmtrash instead,or the full path: /bin/rm'"`
