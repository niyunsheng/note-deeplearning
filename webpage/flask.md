# flask框架

Flask是最轻量的python web框架，仅仅看完了Flask官方文档中的Quickstart部分，就可以实现简单的网站了。当然，网站看起来好不好，与html, CSS和js水平有关，这不在Flask的使用范围之内。

[flask官方教程](https://dormousehole.readthedocs.io/en/latest/)

flask常用命令：
* 安装 `pip install Flask`
* 运行`flask`程序前设置变量`export FLASK_APP=flaskr`
* 调试模式`export FLASK_ENV=development`
  * 如果你打开调试模式，那么服务器会在修改应用代码之后自动重启， 并且当应用出错时还会提供一个有用的调试器。
* 运行`flask`程序`flask run`或者`python -m flask run`
* 设置外部可见`flask run --host=0.0.0.0`

文件结构

```python
/home/user/Projects/flask-tutorial # 项目主目录
├── flaskr/ 
│ ├── __init__.py 
│ ├── db.py 
│ ├── schema.sql 
│ ├── auth.py 
│ ├── blog.py 
│ ├── templates/ 
│ │ ├── base.html 
│ │ ├── auth/ 
│ │ │ ├── login.html 
│ │ │ └── register.html 
│ │ └── blog/ 
│ │ ├── create.html 
│ │ ├── index.html 
│ │ └── update.html 
│ └── static/ 
│ └── style.css 
├── tests/ 
│ ├── conftest.py 
│ ├── data.sql 
│ ├── test_factory.py 
│ ├── test_db.py 
│ ├── test_auth.py 
│ └── test_blog.py 
├── venv/ 
├── setup.py 
└── MANIFEST.in
```

## flask应用工厂

```python
# /flaskr/__init__.py
import os
from flask import Flask
def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True) 
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'), 
    )
    if test_config is None:
    # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)
    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path) 
    except OSError:
        pass
    # a simple page that says hello
    @app.route('/hello') 
    def hello():
        return 'Hello, World!'
    return app
```

1. `app = Flask(__name__, instance_relative_config=True)`创建Flask 实例。
   1. `__name__` 是当前 Python 模块的名称。应用需要知道在哪里设置路径，使用 `__name__` 是一个方 便的方法。
   2. instance_relative_config=True 告诉应用配置文件是相对于instance folder 的相对路径。实例文件夹在 flaskr 包的外面，用于存放本地数据（例如配置密钥和数据库），不应当提交到版本 控制系统。
2. `app.config.from_mapping()` 设置一个应用的缺省配置：
   1. SECRET_KEY 是被 Flask 和扩展用于保证数据安全的。在开发过程中，为了方便可以设置为 'dev' ，但是在发布的时候应当使用一个随机值来重载它。
   2. DATABASE SQLite 数据库文件存放在路径。它位于 Flask 用于存放实例的app.instance_path之内。下一节会更详细地学习数据库的东西。
3. `app.config.from_pyfile()` 使用 `config.py` 中的值来重载缺省配置，如果 config.py 存在的 话。例如，当正式部署的时候，用于设置一个正式的 SECRET_KEY 。
4. `os.makedirs()` 可以确保app.instance_path 存在。Flask 不会自动创建实例文件夹，但是必须确保创建这个文件夹，因为 SQLite 数据库文件会被保存在里面。
5. `@app.route()` 创建一个简单的路由，这样在继续教程下面的内容前你可以先看看应用如何运行的。 它创建了 URL /hello 和一个函数之间的关联。这个函数会返回一个响应，即一个 'Hello, World!'

## 蓝图

略

## 视图

略

## WSGI应用

全称是Web Server Gateway Interface，其主要作用是Web服务器与Python Web应用程序或框架之间的建议标准接口，以促进跨各种Web服务器的Web应用程序可移植性。

WSGI并不是框架而只是一种协议，我们可以将WSGI协议分成三个组件Application，Server，Middleware和协议中传输的内容。

## 最小的flask应用

```python
from flask import Flask 
app = Flask(__name__)

@app.route('/') 
def hello_world():
    return 'Hello, World!'
```

## route装饰器

使用route() 装饰器来把函数绑定到 URL.

## 通过url传递变量

`route`装饰器中还可以设置变量。通过把 URL 的一部分标记为 `<variable_name>` 就可以在 URL 中添加变量。标记的部分会作为关键字参数传递给函数。通过使用 <converter:variable_name> ，可以选择性的加上一个转换器，为变量指定规则。

```python
@app.route('/post/<int:post_id>') 
def show_post(post_id):
    # show the post with the given id, the id is an integer 
    return 'Post %d' % post_id
```

变量的类型有如下五种：

|类型|说明|
|-|-|
|string|（缺省值）接受任何不包含斜杠的文本|
|int|接受正整数|
|float|接受正浮点数|
|path|类似 string ，但可以包含斜杠|
|uuid|接受 UUID 字符串|

## url重定向

```python
@app.route('/projects/') 
def projects():
    return 'The project page'

@app.route('/about') 
def about():
    return 'The about page'
```

projects 的 URL 是中规中矩的，尾部有一个斜杠，看起来就如同一个文件夹。访问一个没有斜杠结尾的 URL 时 Flask 会自动进行重定向，帮你在尾部加上一个斜杠。 

about 的 URL 没有尾部斜杠，因此其行为表现与一个文件类似。如果访问这个 URL 时添加了尾部斜杠就会 得到一个 404 错误。这样可以保持 URL 唯一，并帮助搜索引擎避免重复索引同一页面。

## http方法

http方法公有9种，其中常用的如下：

|方法|描述|
|-|-|
|GET|请求指定的页面信息，并返回实体主体。|
|HEAD|类似于 GET 请求，只不过返回的响应中没有具体的内容，用于获取报头|
|POST|向指定资源提交数据进行处理请求（例如提交表单或者上传文件）。数据被包含在请求体中。POST 请求可能会导致新的资源的建立和/或已有资源的修改。|
|PUT|从客户端向服务器传送的数据取代指定的文档的内容。|
|DELETE|请求服务器删除指定的页面。|

缺省情况下，一个路 由只回应 GET 请求。可以使用route() 装饰器的 methods 参数来处理不同的 HTTP 方法。

`@app.route('/login', methods=['GET', 'POST'])`

## 渲染模板

在python内部生成的html不好阅读，且功能有限，必须自己负责html转义。Falsk自动配置了`Jinja2`模板引擎。

使用render_template() 方法可以渲染模板，你只要提供模板名称和需要作为参数传递给模板的变量就 行了。下面是一个简单的模板渲染例子:

```python
from flask import render_template

@app.route('/hello/') 
@app.route('/hello/<name>') 
def hello(name=None):
    return render_template('hello.html', name=name)
```

Flask 会在 templates 文件夹内寻找模板。因此，如果你的应用是一个模块，那么模板文件夹应该在模块旁 边；如果是一个包，那么就应该在包里面：

情形 1 : 一个模块:

```
/application.py 
/templates 
    /hello.html
```

情形 2 : 一个包:

```
/application
    /__init__.py 
    /templates 
        /hello.html
```

可以充分利用Jinja2模板引擎的威力。

## 操作请求数据

对于 web 应用来说对客户端向服务器发送的数据作出响应很重要。在 Flask 中由全局对象request 来提供请求信息。那么，这个全局变量怎么保证线程安全呢？

实际上，这些对象不是通常意义下的全局对象。这些对象实际上是特定环境下本地对象的代理。

## 接收请求对象

导入`request`对象，然后获取数据。

```python
from flask import request

@app.route('/login', methods=['POST', 'GET'])
def login():
    error = None 
    if request.method == 'POST':
        if valid_login(request.form['username'], 
                        request.form['password']):
            return log_the_user_in(request.form['username'])
        else:
            error = 'Invalid username/password' 
    # the code below is executed if the request method 
    # was GET or the credentials were invalid 
    return render_template('login.html', error=error)
```

当 form 属性中不存在这个键时会发生什么？会引发一个 KeyError.

要操作 URL （如 ?key=value ）中提交的参数可以使用args 属性:

`searchword = request.args.get('key', '')`

## 文件上传

略

## cookies

略

## 会话session

会话相对Cookies更加安全。

`session`对象，允许你在不同请求之间储存信息。这个对象相当于用密钥签名加密的 cookie ，即用户可以查看你的 cookie ，但是如果没有密钥就无法修改它。

使用会话之前你必须设置一个密钥。举例说明:

```python
from flask import Flask, session, redirect, url_for, escape, request

app = Flask(__name__)

# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
@app.route('/')
def index():
    if 'username' in session:
        return 'Logged in as %s' % escape(session['username']) 
    return 'You are not logged in'

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['username'] = request.form['username'] return redirect(url_for('index')) 
    return '''
        <form method="post">
            <p><input type=text name=username> 
            <p><input type=submit value=Login> 
        </form>
        '''
@app.route('/logout') 
def logout():
    # remove the username from the session if it's there
    session.pop('username', None) 
    return redirect(url_for('index'))
```

## 重定向和错误

略

## 日志

略

## 集成WSGI中间件

略

## test_request_context() 环境管理器

```python
from flask import request

with app.test_request_context('/hello', method='POST'):
    # now you can do something with the request until the
    # end of the with block, such as basic assertions: 
    assert request.path == '/hello' 
    assert request.method == 'POST'
```

略