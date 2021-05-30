# html基础语法

`HTML+CSS`是静态网页标准，HTML 指超文本标签语言。CSS 指层叠样式表（Cascading Style Sheets），是用于控制网页样式并允许将样式信息与网页内容分离的一种标记性语言。CSS的引入就是为了使得HTML语言能够更好的适应页面的美工设计，它以HTML语言为基础。

[html参考手册](https://www.w3school.com.cn/tags/index.asp) [css参考手册](https://www.w3school.com.cn/cssref/index.asp)

**学习路径**： 首先学习html和css的基本组件，然后学习html和css联合设计网页。用代码块`<div>`对页面进行布局，每个代码块采用不同的css文件来控制样式。

## 标签和属性

标签： 用`<>`括起来的都是标签
* 标签成对出现`<title>百度一下，你就知道</title>`
* 或者单独出现`<img />`
* 标签有嵌套关系，可以用一颗DOM(Document Object Model（文档对象模型）)树表示

标签的属性
* `<img src="logo.jpg" alt="站标" />`
* 一个标签可能有多个属性 属性先后顺序无关

## html的文件结构

```html
<html>
    <head>
        <title> </title>
    </head>
    <body>

    </body>
</html>
```

* 头部head标签内：浏览器、**搜索引擎所需信息**
* 主体body标签内：网页中包含的具体内容

## html基本标签

html会忽略源代码中的连续空格，会将其变为一个空格，也会忽略换行，将其转化为一个空格。

* 标题 h1-h6
* 段落 p
* 段内换行 `<br />`
* 空格字符 `&nbsp;`
* 预留格式 `<pre> </pre>` 该标签内部的空格和换行会保留
* 行内组合 span ，以便用css对齐格式化
* 水平线 `<hr />` ，默认从网页左端到右端

* 超链接标签 `a`
  * 链接到本网站 `<a href=“news.html”> 新闻 </a>`
  * 其他站点 `<a href=“http://www.baidu.com”> 百度 </a>`
  * 虚拟超链接 `<a href=“#”>版块2</a>`

* 图像标签 `img`
  * 网页常见的图像格式
    * jpg：有损压缩，色彩丰富
    * git：简单动画、背景透明
    * png：无损压缩、透明、交互、动画
  * `<img src="w3school.git" alt="w3c" />`
  * src属性是图片来源，最好采用相对路径，以该文档所在位置为基准
  * slt属性设置图片显示不出来时的替代文本

* 区域列表 div
  * div将网页划分为不同的区域
  * 结合css对不同部分进行样式设计
  * id属性是唯一的

* 无序列表 ul
  * 列表项 li
  * 如导航栏，新闻列表等
* 有序列表 ol
  * 列表项 li
  * 如热搜榜、商品榜等

```html
<ul>
    <li>HTML</li>
    <li>CSS</li>
    <li>JS</li>
</ul>

<ol>
    <li>HTML</li>
    <li>CSS</li>
    <li>JS</li>
</ol>
```

* 表格 table th tr td

```html
<table>
    <tr>
        <td>班级</td>
        <td>学生数</td>
        <td>p平均分</td>
    </tr>
    <tr>
        <td>1班</td>
        <td>30</td>
        <td>89</td>
     </tr>
    <tr>
        <td>2班</td>
        <td>35</td>
        <td>85</td>
    </tr>
    <tr>
        <td>3班</td>
        <td>32</td>
        <td>80</td>
    </tr>
</table>
```

* 表单 form 标签
  * 表单是一个区域，用于采集用户信息
  * 表单元素：包括文本框、按钮、单选、复现、下拉列表、文本域
  * 单选框的name需要相同

```html
<form>
    账户：<input type="text" name="username" /> <br />
    密码：<input type="password" name="userpsd" /> <br />
    <input type="submit" value="提交"/> <br />
    <input type="reset" value="重置"/> <br />
    性别：
    男 <input type="radio" value="boy" name="gender" checked="checked"/>
    女 <input type="radio" value="girl" name="gender"/>  <br />
    爱好：
    音乐 <input type="checkbox" value="1" name="music" checked="checked"/>
    体育 <input type="checkbox" value="2" name="sport" />
    阅读 <input type="checkbox" value="3" name="reading" />  <br />
    省份：
    <select>
        <option>河南</option>
        <option>深圳</option>
        <option selected="selected">北京</option>
    </select> <br />
    个人简介
    <textarea rows="10" cols="50">
        在这里输入内容
    </textarea>
</form>
```

## web语义化

让页面具有良好的结构与 含义，从而让人和机器都 能快速理解网页内容

比如，采用em和strong标签来表示强调（斜体和加粗），而不用i或者b标签

比如，id属性表示该属性唯一，如果不唯一用class属性