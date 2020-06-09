# matplotlib

用`plt.**`和`ax.**`两种画图是有区别的。强烈推荐用ax的方式画图

plt.figure()： plt.***系列。通过http://plt.xxx来画图，其实是取了一个捷径。这是通过matplotlib提供的一个api，这个plt提供了很多基本的function可以让你很快的画出图来，但是如果你想要更细致的精调，就要使用另外一种方法。

`fig, ax = plt.subplots()`
`ax.plot(A,B)`

![](https://matplotlib.org/1.5.1/_images/fig_map.png)

* Figure fig = plt.figure(): 可以解释为画布。
    * 画图的第一件事，就是创建一个画布figure，然后在这个画布上加各种元素。
* Axes ax = fig.add_subplot(1,1,1): 不想定义，没法定义，就叫他axes！
    * 首先，这个不是你画图的xy坐标抽！
    * 可以把axes理解为你要放到画布上的各个物体。比如你要画一个太阳，一个房子，一个车在画布上，那么太阳是一个axes，房子是一个axes，etc。
    * 如果你的figure只有一张图，那么你只有一个axes。如果你的figure有subplot，那么每一个subplot就是一个axes
    * axes是matlibplot的宇宙中心！axes下可以修改编辑的变量非常多，基本上能包含你的所有需求。
* Axis ax.xaxis/ax.yaxis: 对，这才是你的xy坐标轴。
    * 每个坐标轴实际上也是由竖线和数字组成的，每一个竖线其实也是一个axis的subplot，因此ax.xaxis也存在axes这个对象。对这个axes进行编辑就会修改xaxis图像上的表现。

```python
fig, ax = plt.subplots(figsize=(14,7)) # 一个画布
然后在画布上进行各种操作，强烈推荐用ax的方式画图
```

# 基本操作

`import matplotlib.pyplot as plt`

* .plot(x,y)绘制折线图`plt.plot(x,y,format_string,**kwargs)`
    * format_string: 为控制曲线的格式字符串,由 颜色字符、风格字符和标记字符组成。
        * 颜色字符：‘b’蓝色  ；‘#008000’RGB某颜色；‘0.8’灰度值字符串
        * 风格字符：‘-’实线；‘--’破折线； ‘-.’点划线； ‘：’虚线 ； ‘’‘’无线条
        * 标记字符：‘.’点标记  ‘o’ 实心圈 ‘v’倒三角  ‘^’上三角
    * kwargs 第二组或更多的（x, y, format_string）

* `plt.hist(data,bins=10)`绘制直方图,直方图的长方形数目，默认为10
* pyplot文本显示函数
    * `plt.xlabel(‘横轴：时间’, fontproperties = ‘simHei’, fontsize = 20)`
    * `plt.xlabel()`：对x轴增加文本标签
    * `plt.ylabel()`：同理
    * `plt.title()`: 对图形整体增加文本标签
    * `plt.text()`: 在任意位置增加文本
* 中文支持。pyplot并不默认支持中文显示，需要rcParams修改字体来实现
    * rcParams的属性：
        * `font.family` 用于显示字体的名字
        * `font.style` 字体风格，正常’normal’ 或斜体’italic’
        * `font.size` 字体大小，整数字号或者’large’   ‘x-small’

```python
import matplotlib
matplotlib.rcParams[‘font.family’] = ‘STSong’
matplotlib.rcParams[‘font.size’] = 20
# 设定绘制区域的全部字体变成 华文仿宋，字体大小为20
# 中文显示2：只希望在某地方绘制中文字符，不改变别的地方的字体
# 在有中文输出的地方，增加一个属性： fontproperties
plt.xlabel(‘横轴：时间’, fontproperties = ‘simHei’, fontsize = 20)
```

* `.show()`plot方法之后用show方法显示出来
* `legend(['a','b'])`图像标注
* `.imshow()`
* `plt.savefig('fig.png')`

```python
img = cv2.imread('messi5.jpg',0)

import matplotlib.image as imgplt
img = imgplt.imread('Faces/0805personali01.jpg')
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
```

# subplot子图绘制

plt.subplot(3,2,4) :  分成3行2列，共6个绘图区域，在第4个区域绘图。排序为行优先。也可 plt.subplot(324)，将逗号省略。

```python
num_epoch = 0
fig = plt.figure(figsize=(10,10))
for k in range(5*5):
    # i = k // 5
    # j = k % 5
    plt.subplot(5,5,k+1)
    plt.axis('off')
    plt.imshow(imgs[k][0], cmap='gray')
fig.show()
label = 'Epoch {}'.format(num_epoch)
fig.text(0.5, 0.04, label, ha='center')
fig.savefig('{}.png'.format(num_epoch))
```

# 显示动图

```python
import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(-np.pi,np.pi,100)
y=np.sin(x)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x,y)
plt.ion()#使程序plot之后不暂停
plt.show()
plt.pause(1)

for i in range(100):
        try:
                ax.lines.remove(lines[0])
        except Exception:
                pass
        y=np.sin(x+i*0.01)
        lines = ax.plot(x, y, 'r-', lw=1)
        plt.pause(0.1)
```

# [Seaborn的绘图功能](https://blog.csdn.net/kineslave/article/details/82344109)

`import seaborn as sns`

seaborn是matplotlib的高级接口，可以和matplotlib的函数混合使用

* 拟合和绘制回归线或直绘制散点图sns.lmplot()
* 箱图：sns.boxplot(data=?)
* 可视化矩阵内容：sns.heatmap(corr) # 这里用corr相关系数矩阵举例
* 绘制直方图(概率密度)和拟合 sns.distplot()

# 画科研论文图

https://www.zhihu.com/question/21664179/answer/54632841

1. 菜鸟级别：matplotlib
2. 普通级别：绘图语言metapostAsymptote、Asymptote.
3. 图可视化：Graphviz