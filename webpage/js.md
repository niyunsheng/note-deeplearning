# js基础语法

JavaScript 是属于 HTML 和 Web 的编程语言。对网页行为进行编程。

[js参考手册](https://www.w3school.com.cn/js/js_examples.asp)

> 感谢：北京林业大学孙俏老师开设在中国大学MOOC上的[Web前端开发课程](https://www.icourse163.org/course/BFU-1003382003)

JavaScript 是一种运行于JavaScript解释器/引擎中的解释型脚本语言.

注：NodeJS是js的一款本地解释器。浏览器console中可以执行js。

js的组成：
1. 核心(ECMAScript) 
2. 文档对象模型(DOM,Document Object Model),让JS有能力与网页进行对话 
3. 浏览器对象模型(BOM,Browser Object Model) 让JS有能力与浏览器进行对话

js的特点：
* 弱类型语言，由数据来决定数据类型
* 无需编译，直接由 JS引擎负责执行
* 面向对象

浏览器内核的作用
* 负责页面内容的渲染。 内核主要由两部分组成： 
  * 1、内容排版引擎解析 HTML和CSS
  * 2、脚本解释引擎解析 Javascript

## 在网页中嵌入JS脚本

方法1：将JS代码嵌入在元素"事件"中

如：`<button onclick=" console.log( 'Hello World' ); "> 打印消息 </button>`

方法2：将JS代码嵌入在`<script>`标记中

`<script></script>`

方法3：引入外部JS文件

`<script src="js文件路径"></script>`

注意：在body或者head中都可以，推荐放在head中。

## 用法规范

语句
* 允许被JS引擎所解释的代码 
* 使用 **分号** 来表示结束
* 大小写敏感
* 英文标点符号
* 由表达式、关键字、运算符 组成

注释
* 单行注释: `//` 
* 多行注释: `/* */`

## 变量

声明变量时，尽量不要省略`var`关键字，否则声明的是全局变量。

* 定义时不赋值`var userName;`
* 定义时赋值`var bookPrice=25.5;`、`var userName="nys";`

变量名规范
* 不用JS关键字和保留关键字
* 由字母、数字、下划线以及$组成
* 不能以数字开头
* 可以采用驼峰命名法

不能使用未声明的变量，否则报错。

## 数据类型

number类型
* 可以表示32位的整数以及64位的浮点数 
* 整数：32位即4字节 
* 浮点数：即小数，64位，8字节

字符串类型
* 表示一系列的文本字符数据 
* 由Unicode字符，数字，标点组成 
* Unicode 下所有的 字符，数字，标点 在内存中 都占2字节

布尔boolean类型
* 取值只有`true`和`false`
* 做运算时，true可以当做1运算，false当做0运算

undefined类型
* 声明变量未赋值
* 或者是访问对象不存在的属性

和python一样，JS中的类型为弱类型，变量名可以赋值给其他的数据类型。

用`typeof(n)`或者`typeof n`来获取变量n的类型。

用`isNaN(n)`来判断该数据是否为 非数字(Not a Number)

用`n.toString()`获取任意变量n的字符串形式

用`ParseInt(数据)`将数据解析为一个整数，遇到第一个非整数字符即停止

`ParseFloat(数据)`将数据解析为小数，遇到第一个非法字符即停止

## 运算符和表达式

大部分和C++相同，特别的有：
* `==` 等于，不比较类型，只比较数值
* `===` 全等，既比较数值，也比较类型
* `!==` 不全等

## 函数

除了关键字`function`不同，基本和c++相同。

```js
function 函数名(){
    可执行语句;
}
```

JS在正式执行之前，会将所有var声明的变量和function声明的函数，预读到所在作用域的顶部。
但是，对变量的赋值，还保留在原来的位置处。

函数参数传递时，实际上是值传递的方式，不会影响到函数外的变量。

## 分支结构和循环结构

`if-else`的语法和C++完全相同

`switch-case`的语法也和C++完全相同

`while`循环、`do-while`循环、`for`循环

`continue`、`break`

这些的用法均和C++完全相同。

## 数组

类似python中的列表，可以存储任意多个类型不同的元素。

定义数组：
* `var arr1 = [ ];` //定义一个不包含元素的数组 
* `var arr2 = [97, 85, 79];` //定义一个包含3个元素的数组 
* `var arr3 = new Array();` //定义一个不包含元素的数组 
* `var arr4 = new Array(“Tom”, “Mary”, “John”);` //定义一个三个字符串元素的数组

和python的区别在于可以直接用越界的下标赋值元素，数组的length属性会自动变化，length属性的值永远是最大下标+1

减小length属性的值，会删除结尾多余的元素。

数组有三个不限制
* 不限制数组的元素个数:长度可变
* 不限制下标越界
* 不限制元素的数据类型

## 关联数组

关联数组和词典类似，创建方式如下：

```js
var bookInfo = [ ]; 
bookInfo['bookName'] = '西游记' ; 
bookInfo['price'] = 35.5 ;
```

注：由于关联数组的 length 属性值无法获 取其中元素的数量，所以遍历关联数 组只能使用 for..in 循环

```js
for(var key in hash){
    key//只是元素的下标名 
    hash[key]//当前元素值 
}
```

## 数组API

`String(arr)`
* 将arr中每个元素转为字符串，用逗号分隔
* 固定套路: 对数组拍照: 用于鉴别是否数组被修改过

`arr.join(“连接符”)`
* 将arr中每个元素转为字符串，用自定义的连接符分隔
* 将字符组成单词: `chars.join("")`->无缝拼接
* 判断数组是空数组: `arr.join("")==""`
* 将单词组成句子: `words.join(" ")`
* 将数组转化为页面元素的内容`"<开始标签>"+ arr.join("</结束标签><开始标签>") +"</结束标签>"`

`concat()` 
* 拼接两个或更多的数组，并返回结果
* 不直接修改原数组，而返回新数组！
* `var newArr=arr1.concat(值1,值2,arr2,值3,...)`
* 将值1,值2和arr2中每个元素,以及值3都拼接到arr1的元素 之后，返回新数组
* 其中: arr2的元素会被先*打散*，再拼接

`slice()`
* 返回现有数组的一个子数组
* 不直接修改原数组，而返回新数组！
* `var subArr = arr.slice(starti,endi+1)`
* 选取arr中starti位置开始，到endi结束的所有 
* 元素组成新数组返回——原数组保持不变 
* 强调: 含头不含尾
* 复制数组，`arr.slice(0,arr.length);`或简写为`arr.slice();`
* 可以用负数下标
* 一直选取到结尾: 可省略第二个参数

`splice`
* 直接修改原数组
* `arr.splice(starti,n);`
  * 删除arr中starti位置开始的n个元素不考虑含头不含尾 
  * 其实: `var deletes=arr.splice(starti,n);`
  * 返回值deletes保存了被删除的元素组成的临时数组
* `arr.splice(starti,0,值1,值2,...)`
  * 在arr中starti位置，插入新值1,值2,...原starti位置的值 及其之后的值被向后顺移
* `arr.splice(starti,n,值1,值2,...)`
  * 先删除arr中starti位置的n个值，再在starti位置插入新值 
  * 强调: 删除的元素个数和插入的新元素个数不必一致。

`reverse()`
* 颠倒数组中元素的顺序
* 改变原数组
* `arr1.reverse();`

`sort()`
* `arr.sort()`: 默认将所有元素转为字符串再排列
* 问题: 只能排列字符串类型的元素 
* 解决: 使用自定义比较器函数

## DOM

`DOM (document object model)` 是 W3C（万维网联盟）的标准， 是中立于平台和语言的接口，它允许程序和脚本动态地访问和更新文档的内容 、结构和样式。

常见的DOM操作有查找节点、读取节点信息、修改节点信息、创建新节点、删除节点。

|常见的DOM方法|解释|
|-|-|
|getElementById()| |
|getElementsByTagName()| |
|getElementsByClassName()| |
|appendChild()| |
|removeChild()| |
|replaceChild()| |
|insertBefore()| |
|createAttribute()| |
|createElement()| |
|createTextNode()| |
|getAttribute()| |
|setAttribute()| |

### DOM查找

1. 按id属性，精确查找一个元素对象
   1. `var ul = document.getElementById('myList');`
2. 按标签名找
   1. `var elems=parent.getElementsByTagName("tag");`
   2. 查找指定parent节点下的所有标签为tag的子代节点
   3. 可用在任意父元素上
   4. 返回一个动态集合 即使只找到一个元素，也返回集合 必须用[0],取出唯一元素
3. 通过name属性查找
   1. `document.getElementsByName(‘name属性值’)`
   2. 可以返回DOM树中具有指定name属性值的所有子元素集合
4. 通过class查找
   1. 查找父元素下指定class属性的元素
   2. `var elems=parent.getElemnetsByClassName("class");`
5. 通过CSS选择器查找
   1. 只找一个元素`var elem=parent.querySelector("selector")`
      1. selector支持一切css中选择器
      2. 如果选择器匹配的有多个，只返回第一个
   2. 找多个`var elems=parent.querySelectorAll("selector")`
      1. selector API 返回的是非动态集合

### DOM修改

DOM标准有两个，分为核心DOM和HTML DOM，核心DOM可操作一切结构化文档的API, 包括HTML和XML，是万能的，但是比较繁琐。HTML DOM是专门操作HTML文档的简化版DOM API，仅对常用的复杂的API进行了简化，不是万能的，但是简单。

核心DOM的四种操作：
1. 读取属性值
   1. 先获得属性节点对象，再获得节点对象的值:
      1. `var attrNode=elem.attributes[下标/属性名];` 
      2. `var attrNode=elem.getAttributeNode(属性名)`
      3. `attrNode.value`——属性值
   2. 直接获得属性值
      1. `var value=elem.getAttribute("属性名");`
2. 修改属性值
   1. `elem.setAttribute("属性名", value);`
3. 判断是否包含指定属性
   1. `var bool=elem.hasAttribute("属性名")`
4. 移除属性
   1. `elem.removeAttribute("属性名")`

修改css样式表
* 只能修改内联css样式表
* `elem.style.属性名`
* 注意：属性名需要更改：去横线，变驼峰
* 如：background-color => backgroundColor

### DOM添加

添加DOM元素分为三个步骤：
1. 创建空元素
   1. `var elem=document.createElement("元素名")`
2. 设置关键属性
   1. 设置关键属性
      1. 比如：`<a href="http://tmooc.cn">go to tmooc</a>`
      2. `a.innerHTML="go to tmooc"`
      3. `a.herf="http://tmooc.cn";`
   2. 设置关键样式
      1. `a.style.opacity = "1";`
      2. `a.style.cssText = "width: 100px;height: 100px";`
3. 将元素添加到DOM树
   1. 用于为一个父元素追加最后一个子节点`parentNode.appendChild(childNode)`
   2. 用于在父元素中的指定子节点之前添加一个新的子节点`parentNode.insertBefore(newChild, existingChild)`
## BOM

`Browser Object Model`是专门操作浏览器窗口的API——暂时没有标准, 有兼容性问题

浏览器对象模型包括：
* window:代表整个窗口
  * 完整窗口大小: window.outerWidth/outerHeight
  * 文档显示区大小: window.innerWidth/innerHeight
* history:封装当前窗口打开后，成功访问过的历史url记录
* navigator:封装浏览器配置信息
* document:封装当前正在加载的网页内容
* location:封装了当前窗口正在打开的url地址
* screen:封装了屏幕的信息
* event:定义了网页中的事件机制

