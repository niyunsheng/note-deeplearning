# git

本文主要总结参考《Github入门与实践》一书

# git版本控制

版本控制是一种记录一个或若干文件内容变化，以便将来查阅特定版本修订情况的系统。
主要经过了三个阶段：

许多人习惯用复制整个项目目录的方式来保存不同的版本，或许还会改名加上备份时间以示区别。这么做唯一的好处就是简单，但是特别容易犯错。有时候会混淆所在的工作目录，一不小心会写错文件或者覆盖意想外的文件。

1. **本地版本控制系统**。大多都是采用某种简单的数据库来记录文件的历次更新差异。
2. 接下来人们又遇到一个问题，如何让在不同系统上的开发者协同工作？于是， **集中化的版本控制系统**（Centralized Version Control Systems，简称 CVCS）应运而生。这种做法相较于老式的本地 VCS 来说。每个人都可以在一定程度上看到项目中的其他人正在做些什么。 而管理员也可以轻松掌控每个开发者的权限，并且管理一个 CVCS 要远比在各个客户端上维护本地数据库来得轻松容易。缺点是中央服务器的单点故障。如果宕机一小时，那么在这一小时内，谁都无法提交更新，也就无法协同工作。 如果中心数据库所在的磁盘发生损坏，又没有做恰当备份，毫无疑问你将丢失所有数据——包括项目的整个变更历史，只剩下人们在各自机器上保留的单独快照。
3. 于是 **分布式版本控制系统**（Distributed Version Control System，简称 DVCS）面世了。 在这类系统中，像 Git、Mercurial、Bazaar 以及 Darcs 等，客户端并不只提取最新版本的文件快照，而是把代码仓库完整地镜像下来。 这么一来，任何一处协同工作用的服务器发生故障，事后都可以用任何一个镜像出来的本地仓库恢复。 因为每一次的克隆操作，实际上都是一次对代码仓库的完整备份。

总结：git属于分布式版本控制系统，有本地仓库和远程仓库的区别，加入分支的概念，多人可以创建不同的分支以开发新功能，本地测试无误后提交到远程仓库。


# github设计思想

* `pull-request`

全球的开发者通过`pull-request`参与协作。指的是开发者在本地对源码进行更改后，向github中托管的git仓库请求合并的功能。开发者可以在`pull-request`上进行评论交流，比如“修正了bug，可以合并一下吗？”等。通过这个功能，开发者可以轻松的更改源代码，并公开更改的细节，然后向远程仓库提交合并请求。而且，如果请求的更改和项目创始者的意图相违背，也可以选择拒绝合并。

github的`pull-request`不但能轻松查看源代码的前后差别，还可以对制定的一行代码进行评论。通过这样的功能，开发者可以针对具体的代码讨论，使代码审查的工作更加方便。

* 对特定用户进行评论

任务管理和bug报告可以通过issue进行交互。

* 社会化编程的思想

# github相关设置

## watch、star、fork的作用

* 选择watch或者UNwatch就是是否收取对方项目变化的通知
* star类似收藏点赞功能，但是，当你star了过多项目的时候，需要使用插件来按照标签查看这些收藏
* fork就是复制原项目，一般不使用，当你对原项目有改进的时候，可以复制一份自己调试，然后pull、request给原作者，看原作者会不会merge。

## 接收github的消息或邮件提醒

选择watching或者participating时会受到github的站内信和邮件通知。

# git基本操作

## 在本地安装git并配置

直接参考[官网教程](https://git-scm.com/book/zh/v1/%E8%B5%B7%E6%AD%A5-%E5%AE%89%E8%A3%85-Git)
* 直接`sudo apt-get install git`即可安装
* 配置用户名和邮箱
* `git config --global user.name "John Doe"`
* `git config --global user.email johndoe@example.com`
* 查看配置信息`git config --list`

## 设置ssh连接远程仓库

1. 命令行生成秘钥`ssh-keygen -t rsa -C "name@email.com" -f ~/.ssh/id-rsa`

SSH 的全称是Secure Shell，使用非对称加密方式，传输内容使用rsa 或者dsa 加密，可以有效避免网络窃听。有时候，需要能免密码登陆Linux系统，比如Hadoop操作，所以需要开启SSH免密码登陆。

注意，不用的用户生成的密钥是不同的，每个用户生成密钥都放在该用户主目录的”.ssh”目录中；比如：root生成的密钥存放在”/root/.ssh”，个人用户的存放在”/home/[username]/.ssh”目录中

如果A计算机中生成了密钥（id_rsa.pub）。如果在B中执行操作，将A的密钥文件复制到B的”~/.ssh/authorized_keys”文件中。那么，以后A用户使用SSH访问B的时候，就可以免密钥登陆了。这样的好处是，以后机器A生成的密钥分发给很多个机器，这些机器将密钥放入自己的authorized_keys中厚，以后A就可以无密码登陆这些机器了。例如：机器A生成了密钥id_rsa.pub，10.5.110.243将该密钥信息放入authorized_keys文件中，那么机器A以后用SSH访问10.5.110.243便不需要再输入密码了



2. 在github中添加秘钥

/Users/xiahan/.ssh/文件夹下, cat id_rsa.pub。把看到的内容全部复制出来.粘贴到github->setting->SSH key即可

然后就可以用手中的私钥与github进行认证和通信了，输入`ssh -T git@github.com`
如果提示：Hi defnngj You've successfully authenticated, but GitHub does not provide shell access. 说明你连接成功了

ssh只需要连接这一次就可以了，之后可以直接在文件夹里面用push和pull操作。

3. 设置ssh key后push还要输入用户名和密码
因为当前repository使用的是https而不是ssh，需要更新一下origin。
所以，采用git方式下载就不需要这些了

```
git remote rm origin
git remote add origin git@github.com:Username/Your_Repo_Name.git
```

当你输入了这两个命令后，虽然仓库改为了ssh，但是push时会出现Git master branch has no upstream branch错误，根据提示原因是分支数太多，而没有指定分支，用`git push --set-upstream origin master`指定master分支即可。

## 克隆代码库

git支持多种协议，所以有多种方式克隆。

优先采用这种git方式，http方式不支持ssh

`git clone git@github.com:username/hello.git`

`git clone https://github.com/username/username`

如果出现错误信息：`RT ! [rejected] master -> master (fetch first)`。在push远程服务器的时候发现出现此错误；原因是没有同步远程的master。所以我们需要先同步一下`git pull origin master`


## 基本操作
* `git config --global user.name "名称"` 全局设置配置
* `git config user.name` 查看当前配置中用户名
* `git status`查看仓库的状态
* `git add filename`提交文件到暂存区
* `git add .`提交所有改动文件到暂存区
* `git commit -m "comment"`提交暂存区文件到本地仓库并注释
* `git commit`同上，回车后写长注释，git规定每次提交都要写注释
* `git push`将本地仓库提交到远程仓库
* `git log`查看提交日志
* `git diff`查看工作树和暂存区的差别
* `git diff HEAD`仅查看最新提交的差别
* `git pull`获取最新远程仓库的分支
* `git fetch --all && git reset --hard origin/master && git pull`强制覆盖本地代码（与git远程仓库保持一致）(分别执行这三个命令，也可以写在一行)
* `git clone --depth=1`加入depth参数只拉取最新版本，在很多时候，安装代码库时加入此选项，会使下载变快很多
* git的历史信息（.git文件夹）是很难删除的，如果需要一个新的历史，那么，重建git仓库，把当前的全部文件拷过去就行了。

## 版本回溯

1. `git log`找到之前git commit 的id
2. `git reset -hard id`完成撤销，同时将代码恢复到这个commit_id的版本
3. `git reset id`完成Commit命令的撤销，但是不对代码修改进行撤销，可以直接通过git commit 重新提交对本地代码的修改

## 分支操作

* `git branch`查看本地分支
* `git branch -a` 查看远程分支
* `git checkout -b branch_name`创建并切换新分支
* `git checkout branch_name`切换分支
* 将featureA分支合并到master分支，先切换到master分之下，然后合并`git merge --no-ff featureA`
* `git log --graph`以图表形式查看分支
* `git branch -d branch_name`删除分支
* `git push --set-upstream origin branch_name`提交到远程固定分支

## 更改提交的操作

* `git reset commit-id` 恢复到之前的提交

## git全局忽略.DS_Store文件

Mac的每个目录都会有个文件叫做.DS_Store，要解决这个问题，需要配置gitignore文件，但是如果在每个文件夹下都配置该文件，则太过麻烦不实用。问题的解决在于配置全局gitignore文件。

创建文件`vim ~/.gitignore_global`,写上`.DS_Store`,然后用命令`git config --global core.excludesfile /Users/nys/.gitignore_global`配置全局忽略文件即可。

如果要删除GitHub上的文件，直接在网页删除即可。

git也可以设置忽略所有子文件的内容`**/.DS_Stores`，表示忽略所有自文件夹的.DS_Store文件。`*/.DS_Stores`设置忽略当前一级子目录。

## 本地构建git仓库

为了保存代码的方便，防止误操作，所以在本地构建git仓库，并且在关键节点进行add和commit保存镜像，以便回溯代码。

在需要保存的文件夹下面`git init`，然后创建`.gitignore`文件忽略不提交文件，然后`add .`和`commit`保存镜像。

# 下载github中的子文件夹

[参考CSDN](https://blog.csdn.net/u012104219/article/details/79057489)

在SVN里面，这非常容易实现，因为SVN基于文件方式存储，而Git却是基于元数据方式分布式存储文件信息的，它会在每一次Clone的时候将所有信息都取回到本地，即相当于在你的机器上生成一个克隆版的版本库。

1. 将github的url替换成svn的url
  * 比如`https://github.com/NLPIR-team/NLPIR/tree/master/License`,将`tree/master`改为`trunk`
  * 即`https://github.com/NLPIR-team/NLPIR/trunk/License`
2. `svn checkout svnurl`即可

# [SVN](https://www.runoob.com/svn/svn-tutorial.html)

Apache Subversion 通常被缩写成 SVN，是一个开放源代码的版本控制系统.

[官网下载安装](https://www.runoob.com/svn/svn-install.html)

把svn安装目录里的bin目录添加到path路径中，在命令行窗口中输入 svnserve --help ,查看安装正常与否。

获取服务器代码
`svn checkout http://ip/svn/T2000`

更新服务器代码
`svn update`

取消本地修改，还原为服务器版本：
`svn revert [-R] target`

其中target为单个文件时，直接`svn revert target`即可，当target为某个目录时，需要加上-R(Recursive)参数，否则只处理他下面的子目录

添加文件
`svn add file/dir`

将本地更改传输到服务器
`svn commit -m "commit line"`
