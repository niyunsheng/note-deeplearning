# google python styleguide

[zh-google-styleguide](https://github.com/zh-google-styleguide/zh-google-styleguide)

[google-styleguide-pyguide](https://google.github.io/styleguide/pyguide.html)

[toc]

## 使用pylint

tips：使用pylint让代码更规范

pylint是一个在python源码中查找bug的工具，可以捕捉容易忽视的错误，例如输入错误，使用未赋值的变量等。

基本用法：
* 安装`pip install pylint`

Get full report
* `pylint filename.py`

Get errors and warnings
* `pylint -rn filename.py`

Get errors
* `pylint -E filename.py`

Disable warning
* `pylint -rn -d unused-variable filename.py`
* `pylint -rn -d W06012 filename.py`

可以通过`pylint --list-msgs`来获取pylint告警列表。使用`pylint --help-msg=C6409`或者`pylint --help-msg=unused-variable`获取更多特定信息。

## 导入

tips：仅对包和模块使用导入

导入使得命名空间管理十分简单。

导入时不要使用相对名称，即使模块在同一个包中，也要使用完整包名，这可以避免无意间导入一个包两次。

## 包

tips：使用模块的全路径名来导入每个模块

优点是避免模块名冲突，缺点是部署代码时必须赋值包层次。

但是，PEP有不同的解释，可以采用绝对导入和相对导入，优先使用绝对导入。

在[Python Enhancement Proposals 328: Imports: Multi-Line and Absolute/Relative](https://www.python.org/dev/peps/pep-0328/)中，给出了使用绝对路径的理由：
* 随着python库的扩展，很多软件包内部模块无意间掩盖了标准库模块，这种问题很难解决，不能知道作者要使用哪个模块。为了解决歧义，建议采用`sys.path`可以访问的模块或包，即绝对导入。
但是，相对导入也有其理由：
* 如果改变大型程序包的结构，相对导入无序编辑子程序包
* 如果没有相对导入，包中的模块无法轻松的导入自身
* 当前目录用一个点，每上升一级增加一个点

## 异常

Tips：允许使用异常，但是必须小心。

优点：正常操作代码的控制流不会和错误处理代码混在一起. 当某种条件发生时, 它也允许控制流跳过多个框架. 例如, 一步跳出N个嵌套的函数, 而不必继续执行错误的代码.

缺点：可能会导致让人困惑的控制流. 调用库时容易错过错误情况.

使用异常必须遵守特定条件：

1. 像这样触发异常: ``raise MyException("Error message")`` 或者 ``raise MyException`` . 不要使用两个参数的形式( ``raise MyException, "Error message"`` )或者过时的字符串异常( ``raise "Error message"`` ).
2. 模块或包应该定义自己的特定域的异常基类, 这个基类应该从内建的Exception类继承. 模块的异常基类应该叫做"Error".

```python
    class Error(Exception):
        pass
```

3. 永远不要使用 ``except:`` 语句来捕获所有异常, 也不要捕获 ``Exception`` 或者 ``StandardError`` , 除非你打算重新触发该异常, 或者你已经在当前线程的最外层(记得还是要打印一条错误消息). 在异常这方面, Python非常宽容, ``except:`` 真的会捕获包括Python语法错误在内的任何错误. 使用 ``except:`` 很容易隐藏真正的bug. 
4. 尽量减少try/except块中的代码量. try块的体积越大, 期望之外的异常就越容易被触发. 这种情况下, try/except块将隐藏真正的错误. 
5. 使用finally子句来执行那些无论try块中有没有异常都应该被执行的代码. 这对于清理资源常常很有用, 例如关闭文件.
6. 当捕获异常时, 使用 ``as`` 而不要用逗号. 例如

```python
try:
    raise Error
except Error as error:
    pass
```

## 全局变量

tips：避免全局变量

缺点：导入模块时会对模块级变量赋值，可能会改变模块行为。

避免使用全局变量，用类变量的代替，但是也有例外：
* 脚本的默认选项.
* 模块级常量. 例如:　PI = 3.14159. 常量应该全大写, 用下划线连接.
* 有时候用全局变量来缓存值或者作为函数返回值很有用.
* 如果需要, 全局变量应该仅在模块内部可用, 并通过模块级的公共函数来访问.

## 嵌套/局部/内部类或函数

tip：鼓励使用嵌套/本地/内部类或函数，用于限制工具类和函数的有效范围

## 列表推导

tips：简单的情况下用列表推导，但是避免特别复杂的列表推导，因为难以阅读。

## 默认迭代器和操作符

tips：如果类型支持，就是用默认的迭代器和操作符(in和not in)。

## 生成器

tips：鼓励使用，没有缺点。

## lambda函数

tips：适用于单行函数

lambda常用语为map和filter之类的高阶函数定义回调函数或者操作符。

如果代码超过了60-80个字符，最好还是定义成常规(嵌套)函数。

## 条件表达式

`x = 1 if cond else 2`

tips：适用于单行函数

## 默认参数值

tips：适用于大部分情况

python不支持重载方法和函数，默认参数是一种仿造重载行为的简单方式。

## 使用properties装饰器

访问和设置数据成员时，通常会使用轻量级的访问和设置函数，建议用properties来代替。通过消除对简单属性访问的显式get和set方法调用，提高了可读性。允许计算是懒惰的。

```python
class Square:
    '''
    To use:
    >>> sq = Square(3)
    >>> sq.area
    9
    '''
    def __init__(self, side):
        self.side = side

    @property
    def area(self):
        """Area of the square."""
        return self._get_area()
```

## True/False的求值

所有的空被认为是false，因此0, None, [], {}都被认为是false。

* 当你写`if x`时，实际是`if x is not None`
* 不要用==比较布尔量，应该用`if not x`或者`if x`

## 过时的语言特性

tips：尽可能使用字符串方法取代字符串模块. 使用函数调用语法取代apply(). 使用列表推导, for循环取代filter(), map()以及reduce().

## 词法作用域(Lexical Scoping)

tips：推荐使用

嵌套的Python函数可以引用外层函数中定义的变量, 但是不能够对它们赋值. 变量绑定的解析是使用词法作用域, 也就是基于静态的程序文本. 对一个块中的某个名称的任何赋值都会导致Python将对该名称的全部引用当做局部变量, 甚至是赋值前的处理. 

```python
def get_adder(summand1):
    """Returns a function that adds numbers to a given number."""
    def adder(summand2):
        return summand1 + summand2
    return adder
```

## 函数与方法装饰器

tips：如果好处很显然, 就明智而谨慎的使用装饰器

优点:优雅的在函数上指定一些转换. 该转换可能减少一些重复代码, 保持已有函数不变(enforce invariants), 等.

比如计算函数运行时间的装饰器：

```python
import time
def clock(func):
    '''
    To use:
    @clock
    def fun():
        pass
    '''
    def clocked(*args):
        t0 = time.time()
        result = func(*args)
        elapsed = time.time() - t0
        name = func.__name__
        arg_str = ', '.join(repr(arg) for arg in args)
        print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
        return result
    return clocked
```

## 线程

tips：不要依赖内建类型的原子性.

优先使用Queue模块的 Queue 数据类型作为线程间的数据通信方式. 另外, 使用threading模块及其锁原语(locking primitives). 了解条件变量的合适使用方式, 这样你就可以使用 threading.Condition 来取代低级别的锁了.

## 威力过大的特性

tips：在你的代码中避免这些特性.

Python是一种异常灵活的语言, 它为你提供了很多花哨的特性, 诸如元类(metaclasses), 字节码访问, 任意编译(on-the-fly compilation), 动态继承, 对象父类重定义(object reparenting), 导入黑客(import hacks), 反射, 系统内修改(modification of system internals), 等等.

# python风格规范

## 分号

不要在行尾加分号, 也不要用分号将两条命令放在同一行.

## 行长度

每行不超过80个字符。

不要使用反斜杠连接行。Python会将 圆括号, 中括号和花括号中的行隐式的连接起来 , 你可以利用这个特点. 如果需要, 你可以在表达式外围增加一对额外的圆括号,这时应该垂直对齐换行的元素。

```python
foo_bar(self, width, height, color='black', design=None, x='foo',
             emphasis=None, highlight=0)
if (width == 0 and height == 0 and
         color == 'red' and emphasis == 'strong'):
    pass
x = ('This will build a very long long '
     'long long long long long long string')
```

在注释中，如果必要，将长的URL放在一行上，不要因为长放两行

## 括号

除非是用于实现行连接, 否则不要在返回语句或条件语句中使用括号. 不过在元组两边使用括号是可以的.

## 缩进

用4个空格来缩进代码。绝对不要用tab, 也不要tab和空格混用。

函数参数过长导致的换行，要么换行后对齐参数，要么第一行不写参数，参数都写在第二行，缩进4空格。

```python
# Aligned with opening delimiter
foo = long_function_name(var_one, var_two,
                        var_three, var_four)

# Aligned with opening delimiter in a dictionary
foo = {
    long_dictionary_key: value1 +
                         value2,
    ...
}

# 4-space hanging indent; nothing on first line
foo = long_function_name(
    var_one, var_two, var_three,
    var_four)

# 4-space hanging indent in a dictionary
foo = {
    long_dictionary_key:
        long_dictionary_value,
    ...
}
```

## 空行

顶级定义（函数或者类）之间空两行, 方法定义之间空一行。

## 空格

按照标准的排版规范来使用标点两边的空格。

* 括号、方括号、大括号内不要有空格.

`spam(ham[1], {eggs: 2}, [])`

* 逗号，分号或冒号前没有空格，后面需要加空格，除非是在结尾

```python
if x == 4:
    print(x, y)
x, y = y, x
```

* 参数列表, 索引或切片的左括号前不应加空格.

`spam(1)`

`dict['key'] = list[index]`

* 在二元操作符两边都加上一个空格, 比如赋值(=), 比较(==, <, >, !=, <>, <=, >=, in, not in, is, is not), 布尔(and, or, not).

`x == 1`

* 当’=’用于指示关键字参数或默认参数值时, 不要在其两侧使用空格.

`def complex(real, imag=0.0): return magic(r=real, i=imag)`

* 不要用空格来垂直对齐多行间的标记, 因为这会成为维护的负担(适用于:, #, =等):

```python
Yes:
    foo = 1000  # comment
    long_name = 2  # comment that should not be aligned

    dictionary = {
        "foo": 1,
        "long_name": 2,
        }
No:
    foo       = 1000  # comment
    long_name = 2     # comment that should not be aligned

    dictionary = {
        "foo"      : 1,
        "long_name": 2,
        }
```


## Shebang

根据PEP-394，程序的main文件应该以 #!/usr/bin/python2或者 #!/usr/bin/python3开始。

在文件中存在Shebang的情况下, 类Unix操作系统的程序载入器会分析Shebang后的内容, 将这些内容作为解释器指令, 并调用该指令, 并将载有Shebang的文件路径作为该解释器的参数. 例如, 以指令#!/bin/sh开头的文件在执行时会实际调用/bin/sh程序.

`#!`先用于帮助内核找到Python解释器, 但是在导入模块时, 将会被忽略. 因此只有被直接执行的文件中才有必要加入。

## 注释

### Docstrings

Python有一种独一无二的的注释方式: 使用**文档字符串**. 文档字符串是包, 模块, 类或函数里的第一个语句. 这些字符串可以通过对象的__doc__成员被自动提取, 并且被pydoc所用。

我们对文档字符串的惯例是使用三重双引号"""( PEP-257 ). 一个文档字符串应该这样组织: 首先是一行以句号, 问号或惊叹号结尾的概述(或者该文档字符串单纯只有一行). 接着是一个空行. 接着是文档字符串剩下的部分, 它应该与文档字符串的第一行的第一个引号对齐. 

每个文件应该包含一个许可样板. 根据项目使用的许可(例如, Apache 2.0, BSD, LGPL, GPL), 选择合适的样板.

每个文件开头都应该包含该模块的用法

```python
"""A one line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

  Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
```

### Modules

这里所指的函数,包括函数, 方法, 以及生成器。一个函数必须要有文档字符串, 除非它满足以下条件:
* 外部不可见
* 非常短小
* 简单明了

文档字符串应该包含函数做什么, 以及输入和输出的详细描述. 通常, 不应该描述”怎么做”, 除非是一些复杂的算法. 文档字符串应该提供足够的信息, 当别人编写代码调用该函数时, 他不需要看一行代码, 只要看文档字符串就可以了. 对于复杂的代码, 在代码旁边加注释会比使用文档字符串更有意义.

Args:列出每个参数的名字, 并在名字后使用一个冒号和一个空格, 分隔对该参数的描述.如果描述太长超过了单行80字符,使用2或者4个空格的悬挂缩进(与文件其他部分保持一致). 描述应该包括所需的类型和含义. 如果一个函数接受`*foo`(可变长度参数列表)或者`**bar` (任意关键字参数), 应该详细列出`*foo`和`**bar`.

Returns: (或者 Yields: 用于生成器)
描述返回值的类型和语义. 如果函数返回None, 这一部分可以省略.

Raises:列出与接口有关的所有异常.

```python
def fetch_smalltable_rows(table_handle: smalltable.Table,
                          keys: Sequence[Union[bytes, str]],
                          require_all_keys: bool = False,
                         ) -> Mapping[bytes, Tuple[str]]:
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        table_handle: An open smalltable.Table instance.
        keys: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.
        require_all_keys: Optional; If require_all_keys is True only
          rows with values set for all keys will be returned.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {b'Serak': ('Rigel VII', 'Preparer'),
         b'Zim': ('Irk', 'Invader'),
         b'Lrrr': ('Omicron Persei 8', 'Emperor')}

        Returned keys are always bytes.  If a key from the keys argument is
        missing from the dictionary, then that row was not found in the
        table (and require_all_keys must have been False).

    Raises:
        IOError: An error occurred accessing the smalltable.
    """
```

在args后面换行空两格也是可以的

```python
def fetch_smalltable_rows(table_handle: smalltable.Table,
                          keys: Sequence[Union[bytes, str]],
                          require_all_keys: bool = False,
                         ) -> Mapping[bytes, Tuple[str]]:
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
      table_handle:
        An open smalltable.Table instance.
      keys:
        A sequence of strings representing the key of each table row to
        fetch.  String keys will be UTF-8 encoded.
      require_all_keys:
        Optional; If require_all_keys is True only rows with values set
        for all keys will be returned.

    Returns:
      A dict mapping keys to the corresponding table row data
      fetched. Each row is represented as a tuple of strings. For
      example:

      {b'Serak': ('Rigel VII', 'Preparer'),
       b'Zim': ('Irk', 'Invader'),
       b'Lrrr': ('Omicron Persei 8', 'Emperor')}

      Returned keys are always bytes.  If a key from the keys argument is
      missing from the dictionary, then that row was not found in the
      table (and require_all_keys must have been False).

    Raises:
      IOError: An error occurred accessing the smalltable.
    """
```

### 类

类应该在其定义下有一个用于描述该类的文档字符串. 如果你的类有公共属性(Attributes), 那么文档中应该有一个属性(Attributes)段. 并且应该遵守和函数参数相同的格式.

```python
class SampleClass:
    """Summary of class here.

    Longer class information....
    Longer class information....

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    def __init__(self, likes_spam=False):
        """Inits SampleClass with blah."""
        self.likes_spam = likes_spam
        self.eggs = 0

    def public_method(self):
        """Performs operation blah."""
```

### 块注释和行注释

最需要写注释的是代码中那些技巧性的部分. 如果你在下次 代码审查 的时候必须解释一下, 那么你应该现在就给它写注释. 对于复杂的操作, 应该在其操作开始前写上若干行注释. 对于不是一目了然的代码, 应在其行尾添加注释.

```python
# We use a weighted dictionary search to find out where i is in
# the array.  We extrapolate position based on the largest num
# in the array and the array size and then do binary search to
# get the exact number.

if i & (i-1) == 0:  # True if i is 0 or a power of 2.
```

## 类

如果一个类不继承自其它类, 就显式的从object继承. 嵌套类也一样.

```python
class SampleClass(object):
         pass


class OuterClass(object):

    class InnerClass(object):
        pass


class ChildClass(ParentClass):
    """Explicitly inherits from another class already."""
```

继承自 object 是为了使属性(properties)正常工作, 并且这样可以保护你的代码, 使其不受 PEP-3000 的一个特殊的潜在不兼容性影响.

## 字符串

使用`format`方法或`%`运算符来格式化字符串，即使参数都是字符串。尽您最大的判断力在`+`和 `%`（或`format`）之间做出选择。

```python
x = a + b
x = '%s, %s!' % (imperative, expletive)
x = '{}, {}'.format(first, second)
x = 'name: %s; score: %d' % (name, n)
x = 'name: {}; score: {}'.format(name, n)
x = f'name: {name}; score: {n}'  # Python 3.6+
```

避免使用`+=`运算符在循环内累积字符串。由于字符串是不可变的，因此会创建不必要的临时对象，并导致二次而不是线性运行时间。而是，将每个子字符串添加到列表中，`''.join`然后在循环终止后将该列表添加到列表中（或将每个子字符串写入`io.BytesIO`缓冲区中）。

```python
items = ['<table>']
for last_name, first_name in employee_list:
    items.append('<tr><td>%s, %s</td></tr>' % (last_name, first_name))
items.append('</table>')
employee_table = ''.join(items)
```

与您在文件中选择的字符串引号字符一致。选择`'` 或`"`坚持下去。可以在字符串上使用其他引号字符，以避免`\\` 在字符串内进行转义。

首选`"""`多行字符串，而不是`'''`。Docstrings必须使用`"""`

多行字符串不随程序其余部分的缩进一起流动。如果需要避免在字符串中嵌入多余的空格，请使用串联的单行字符串或多行字符串with textwrap.dedent() 删除每行的初始空格：

```python
long_string = """This is fine if your use case can accept
    extraneous leading spaces."""

long_string = ("And this is fine if you cannot accept\n" +
                "extraneous leading spaces.")

long_string = ("And this too is fine if you cannot accept\n"
                "extraneous leading spaces.")

import textwrap

long_string = textwrap.dedent("""\
    This is also fine, because textwrap.dedent()
    will collapse common leading spaces in each line.""")
```

## 文件和sockets

在文件和sockets结束时, 显式的关闭它.推荐使用with语句管理文件。对于不支持使用”with”语句的类似文件的对象,使用 contextlib.closing():

```python

with contextlib.closing(urllib.urlopen("http://www.python.org/")) as front_page:
    for line in front_page:
        print line
```

## TODO注释

TODO对于临时的，短期的解决方案或足够好但不完美的代码，请使用注释。

TODO注释应该在所有开头处包含”TODO”字符串, 紧跟着是用括号括起来的你的名字, email地址或其它标识符. 然后是一个可选的冒号. 接着必须有一行注释, 解释要做什么. 主要目的是为了有一个统一的TODO格式, 这样添加注释的人就可以搜索到(并可以按需提供更多细节). 写了TODO注释并不保证写的人会亲自解决问题. 当你写了一个TODO, 请注上你的名字.

```python
# TODO(kl@gmail.com): Use a "*" here for string repetition.
# TODO(Zeke) Change this to use relations.
```

## 导入

每个导入应该独占一行。

导入总应该放在文件顶部, 位于模块注释和文档字符串之后, 模块全局变量和常量之前. 导入应该按照从最通用到最不通用的顺序分组:
* 标准库导入
* 第三方库导入
* 应用程序指定导入

每种分组中, 应该根据每个模块的完整包路径按字典序排序, 忽略大小写.

```python
import foo
from foo import bar
from foo.bar import baz
from foo.bar import Quux
from Foob import ar
```

## 语句

通常每个语句应该独占一行。不过, 如果测试结果与测试语句在一行放得下, 你也可以将它们放在同一行. 如果是if语句, 只有在没有else时才能这样做. 

## 访问控制

在Python中, 对于琐碎又不太重要的访问函数, 你应该直接使用公有变量来取代它们, 这样可以避免额外的函数调用开销. 当添加更多功能时, 你可以用属性(property)来保持语法的一致性.
> 虽然在OOP中，一直被教育所有成员变量都必须是私有的! 其实, 那真的是有点麻烦啊. 试着去接受Pythonic哲学吧

## 命名

应该避免的名称：
* 单字符名称, 除了计数器和迭代器.
* 包/模块名中的连字符(-)
* 双下划线开头并结尾的名称(Python保留, 例如__init__)

命名约定
* 所谓”内部(Internal)”表示仅模块内可用, 或者, 在类内是保护或私有的.
* 用单下划线(_)开头表示模块变量或函数是protected的(使用from module import *时不会包含).
* 用双下划线(__)开头的实例变量或方法表示类内私有.
* 将相关的类和顶级函数放在同一个模块里. 不像Java, 没必要限制一个类一个模块.
* 对类名使用大写字母开头的单词(如CapWords, 即Pascal风格), 但是模块名应该用小写加下划线的方式(如lower_with_under.py). 尽管已经有很多现存的模块使用类似于CapWords.py这样的命名, 但现在已经不鼓励这样做, 因为如果模块名碰巧和类名一致, 这会让人困扰.

## Main

即使是一个打算被用作脚本的文件, 也应该是可导入的. 并且简单的导入不应该导致这个脚本的主功能(main functionality)被执行, 这是一种副作用. 主功能应该放在一个`main()`函数中.

```python
def main():
      ...

if __name__ == '__main__':
    main()
```