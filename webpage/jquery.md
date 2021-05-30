# JQuery

> 感谢：北京林业大学孙俏老师开设在中国大学MOOC上的[Web前端开发课程](https://www.icourse163.org/course/BFU-1003382003)

JQuery是快速、简洁的第三方js库。

核心理念是write less，do more（写更少的代码，做更多的事情）

使用JQuery有两个优点，首先是对DOM操作的终极简化，其次是屏蔽了浏览器的兼容性问题。

使用JQuery有两种方式，第一种是将JQuery.js 下载到服务器本地，在script中使用服务器路径；第二种是使用CDN网络上共享的JQuery.js，生产环境中用的最多。

## 工厂函数`$()`

在JQuery中，无论使用哪种类型的选择符，都要从一个美 元符号`$`和一对圆括号开始：`$()`

所有能在样式表中使用的选择符，都能放到这个圆括号中的引号内。

对比：
```js
//DOM:
    document.getElementById('myList');
//Jquery:
    $("#myList");
```

## JQuery增删改查操作

### 查找

1. 基本选择器
   1. `#id` `.class` 同CSS
2. 层级选择器
   1. 后代选择器 子代选择器 同CSS
3. 兄弟关系
   1. `$("...").next/prev()` 紧邻的前一个或者后一个元素
   2. `$("...").nextAll/prevAll()` 之前或者之后的所有元素
   3. `$("...").siblings()`除自己之外的所有兄弟

### 修改

元素属性修改
1. 获取`$("...").attr("属性名")`
2. 修改`$("...").attr("属性名",值)`

内容修改
1. html操作
   1. `html( )`：读取或修改节点的HTML内容
   2. 获取`<p>`元素的HTML代码`$("p").html()`
   3. 设置`<p>`元素的HTML代码`$("p").html("<strong>你最喜欢的水果是?</strong>");`
2. 文本操作
   1. 获取`<p>`元素的文本`$("p").text()`
   2. 设置`<p>`元素的文本`$("p").text("你最喜欢的水果是?");`
3. 值操作
   1. `val( )`：读取或修改节点的value属性值
   2. 获取按钮的value值`$("input:eq(5)").val();`
   3. 设置按钮的value值`$("input").val("我被点击了!");`

样式修改
1. 直接修改css属性
   1. 获取css样式(计算后的样式)`$("...").css("CSS属性名")`
   2. 修改css样式`$("...").css("css属性名"，值)`
2. 通过修改class批量修改样式
   1. 判断是否包含指定class`$("...").hasClass("类名")`
   2. 添加class`$("...").addClass("类名")`
   3. 移除class`$("...").removeClass("类名")`

### 添加和删除

添加
1. 创建新元素`var $new = $("html代码片段")`
2. 将新元素结尾添加到DOM树`$(parent).append($newelem)`

删除
1. `$("...").remove()`
2. 获取第二个`<li>`元素节点后，将它从网页中删除`$("ul li:eq(1)").remove();`
3. 把`<li>`元素中属性title不等于"菠萝"的`<li>`元素删除`$("ul li").remove("li[title!=菠萝]");`

## JQuery事件

事件绑定

语法：`$("...").bind("事件类型"，function(e){....})`

如`$("...").bind("click"，function(e){....})`，也可以简写为`$("...").click(function(e){....})`

事件对象

```js
$("#btn").click(function(e){
    console.log("hello"); 
})
```

这个对象中包含与事件相关的信息，也提供了可以影响事件在DOM中传递进程的一些方法。

事件对象记录事件发生时的鼠标位置、键盘按键状态和触发对象等信息
* `clientX/offsetX/pageX/screenX/x`：事件发生的X坐标 
* `clientY/offsetY/pageY/screenY/y`：事件发生的Y坐标 
* `keyCode` : 键盘事件中按下的按键的值