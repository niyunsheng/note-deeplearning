# python 进程、线程、协程、通信、同步互斥

[廖雪峰基础知识讲解](https://www.liaoxuefeng.com/wiki/1016959663602400/1017627212385376)

基础工具函数：
* 获取当前进程的pid：`os.getpid()`
* 获取父进程的pid：`os.getppid()`

父进程通过`fork()`系统调用可以复制出一个子进程来处理新任务，常见的Apache服务器就是由父进程监听端口，每当有新的http请求时，就fork出子进程来处理新的http请求。


## 进程

单子进程用`multiprocessing.Process`
* `start()` 子进程开始执行
* `join()` 同步机制，主进程等待该子进程结束
* `terminate()` 强制杀死子进程

多个同样任务的子进程用进程池`multiprocessing.Pool`
* `close()` 关闭进程池，不再接收新任务
* `join()` 等待所有子进程结束， join方法要在close或terminate之后使用
* `terminate()` 结束工作进程，不在处理未处理的任务
* `results = apply(func[, args[, kwds]])` 阻塞，一个子进程执行完毕前，主进程会阻塞
* `results = apply_async(func[, args[, kwds[, callback]]])` 前者的非阻塞版本，但是如果调用`results.get()`来获取结果回调，则会阻塞
* `map(func, iterable[, chunksize])` 
  * map方法与内置的map函数行为基本一致，在它会使进程阻塞与此直到结果返回
  * 但需注意的是其第二个参数虽然描述的为iterable, 但在实际使用中发现只有在整个队列全部就绪后，程序才会运行子进程。
* `map_async(func, iterable[, chunksize[, callback]])`
  * 与map用法一致，但是它是非阻塞的。其有关事项见apply_async。
* `imap(func, iterable[, chunksize])`
  * 与map不同的是， imap的返回结果为iter，需要在主进程中主动使用next来驱动子进程的调用。即使子进程没有返回结果，主进程对于gen_list(l)的 iter还是会继续进行， 另外根据python2.6文档的描述，对于大数据量的iterable而言，将chunksize设置大一些比默认的1要好。
* `imap_unordered(func, iterable[, chunksize])`
  * 同imap一致，只不过其并不保证返回结果与迭代传入的顺序一致。

> 如果想要执行一个其他的命令行进程，如`cp a.py b.py`，可以用`subprocess`包。

进程通信，可以用`from multiprocessing import Queue, Pipes`来实现。

多进程生产者消费者举例：

```python
from multiprocessing import Process, Queue
import os, time, random

# 写数据进程执行的代码:
def write(q, data):
    print('Process to write: %s' % os.getpid())
    for value in data:
        print('{} :Put {} to queue...'.format(os.getpid(), value))
        q.put(value)
        time.sleep(random.random())

# 读数据进程执行的代码:
def read(q):
    print('Process to read: %s' % os.getpid())
    while True:
        value = q.get(True)
        print('{} :Get {} from queue.'.format(os.getpid(), value))

if __name__=='__main__':
    # 父进程创建Queue，并传给各个子进程：
    q = Queue()
    pw1 = Process(target=write, args=(q,[1,2,3]))
    pw2 = Process(target=write, args=(q,[4,5,6]))
    pr1 = Process(target=read, args=(q,))
    pr2 = Process(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw1.start()
    pw2.start()
    # 启动子进程pr，读取:
    pr1.start()
    pr2.start()
    # 等待pw结束:
    pw1.join()
    pw2.join()
    # pr进程里是死循环，无法等待其结束，只能强行终止:
    pr1.terminate()
    pr2.terminate()
```

输出结果(不唯一)：

```
Process to write: 683947
683947 :Put 1 to queue...
Process to write: 683948
683948 :Put 4 to queue...
Process to read: 683949
683949 :Get 1 from queue.
683949 :Get 4 from queue.
Process to read: 683951
683947 :Put 2 to queue...
683949 :Get 2 from queue.
683948 :Put 5 to queue...
683951 :Get 5 from queue.
683947 :Put 3 to queue...
683949 :Get 3 from queue.
683948 :Put 6 to queue...
683951 :Get 6 from queue.
```

进程池：
* 如果需要父进程和子进程通信，则不能用`multiprocessing.Queue()`，而需要用`multiprocessing.Manager().Queue()`

```python
from multiprocessing import Pool, Queue, Manager
import os, time, random

# 写数据进程执行的代码:
def write(q, data):
    print('Process to write: %s' % os.getpid())
    for value in data:
        print('{} :Put {} to queue...'.format(os.getpid(), value))
        q.put(value)
        time.sleep(random.random())
    return data

# 读数据进程执行的代码:
def read(q):
    print('Process to read: %s' % os.getpid())
    while True:
        value = q.get(True)
        print('{} :Get {} from queue.'.format(os.getpid(), value))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(2)
    q = Manager().Queue()
    for i in range(5):
        p.apply_async(write, args=(q,[i]))
    for i in range(5):
        p.apply_async(read, args=(q,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    p.terminate()
    print('All subprocesses done.')
```

结果是（不唯一）：

```
Parent process 1696274.
Waiting for all subprocesses done...
Process to write: 1696276
1696276 :Put 1 to queue...
Process to write: 1696275
1696275 :Put 0 to queue...
Process to write: 1696276
1696276 :Put 2 to queue...
Process to write: 1696276
1696276 :Put 3 to queue...
Process to write: 1696275
1696275 :Put 4 to queue...
Process to read: 1696275
1696275 :Get 1 from queue.
1696275 :Get 0 from queue.
1696275 :Get 2 from queue.
1696275 :Get 3 from queue.
1696275 :Get 4 from queue.
Process to read: 1696276
```



## 线程

用`threading`包，实现了对线程的高级封装。
* `threading.current_thread().name` 获取当前线程的名字


多进程通信用队列，多线程通信用共享变量即可，共享变量的读取加锁`threading.Lock()`即可。

用`queue.Queue()`实现线程通信（**这个队列是线程安全的**），没有`.terminate()`方法，需要在线程中设置异常或者设置对全局变量的判断自己主动结束。

```python
import threading
import os, time, random
from queue import Queue

# 写数据进程执行的代码:
def write(q, data):
    print('Process to write: %s' % os.getpid())
    for value in data:
        print('{} :Put {} to queue...'.format(threading.current_thread().name, value))
        q.put(value)
        time.sleep(random.random())

# 读数据进程执行的代码:
def read(q):
    print('Process to read: %s' % os.getpid())
    while True:
        value = q.get(True)
        print('{} :Get {} from queue.'.format(threading.current_thread().name, value))

if __name__=='__main__':
    # 父进程创建Queue，并传给各个子进程：
    q = Queue()
    pw1 = threading.Thread(target=write, args=(q,[1,2,3]))
    pw2 = threading.Thread(target=write, args=(q,[4,5,6]))
    pr1 = threading.Thread(target=read, args=(q,))
    pr2 = threading.Thread(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw1.start()
    pw2.start()
    # 启动子进程pr，读取:
    pr1.start()
    pr2.start()
    # 等待pw结束:
    pw1.join()
    pw2.join()
```

### 对多线程的GIL的进一步解释

Python的线程是真正的Posix Thread，而不是模拟出来的线程。但解释器执行代码时，有一个GIL锁：Global Interpreter Lock，任何Python线程执行前，必须先获得GIL锁，然后，每执行100条字节码，解释器就自动释放GIL锁，让别的线程有机会执行。这个GIL全局锁实际上把所有线程的执行代码都给上了锁，所以，多线程在Python中只能交替执行，即使100个线程跑在100核CPU上，也只能用到1个核。（**即所有的核都用到，但是每个核的利用率为1%**）

```python
import threading, multiprocessing

def loop():
    x = 0
    while True:
        x = x ^ 1

for i in range(multiprocessing.cpu_count()):
    t = threading.Thread(target=loop)
    t.start()
```

## 分布式进程

[参考](https://www.liaoxuefeng.com/wiki/1016959663602400/1017631559645600)

在Thread和Process中，应当优选Process，因为Process更稳定，而且，Process可以分布到多台机器上，而Thread最多只能分布到同一台机器的多个CPU上。

Python的multiprocessing模块不但支持多进程，其中managers子模块还支持把多进程分布到多台机器上。一个服务进程可以作为调度者，将任务分布到其他多个进程中，依靠网络通信。由于managers模块封装很好，不必了解网络通信的细节，就可以很容易地编写分布式多进程程序。

## WSGI接口

在写python的web应用时，正确的做法是底层代码由专门的服务器软件实现，我们用Python专注于生成HTML文档。因为我们不希望接触到TCP连接、HTTP原始请求和响应格式，所以，需要一个统一的接口，让我们专心用Python编写Web业务。

这个接口就是WSGI：Web Server Gateway Interface。

## 协程

协程类似与多线程，多个协程之间可以由主程序来控制中断和切换，而线程的切换时cpu控制的，协程也不需要线程切换的开销。

其次，协程不需要用锁机制，因为只有一个线程，也不存在同时写变量冲突，在协程中控制共享资源不加锁，只需要判断状态就好了，所以执行效率比多线程高很多。

协程是在单个进程中的， 如果想要用多核的效率，应该采用多进程+协程的机制。

协程和生成器有相似，但是不一样，如下例：

```python
def consumer():
    r = ''
    while True:
        n = yield r
        if not n:
            return
        print('[CONSUMER] Consuming %s...' % n)
        r = '200 OK'

def produce(c):
    c.send(None)
    n = 0
    while n < 5:
        n = n + 1
        print('[PRODUCER] Producing %s...' % n)
        r = c.send(n)
        print('[PRODUCER] Consumer return: %s' % r)
    c.close()

c = consumer()
produce(c)
```

结果是：
```
[PRODUCER] Producing 1...
[CONSUMER] Consuming 1...
[PRODUCER] Consumer return: 200 OK
[PRODUCER] Producing 2...
[CONSUMER] Consuming 2...
[PRODUCER] Consumer return: 200 OK
[PRODUCER] Producing 3...
[CONSUMER] Consuming 3...
[PRODUCER] Consumer return: 200 OK
[PRODUCER] Producing 4...
[CONSUMER] Consuming 4...
[PRODUCER] Consumer return: 200 OK
[PRODUCER] Producing 5...
[CONSUMER] Consuming 5...
[PRODUCER] Consumer return: 200 OK
```

最后套用Donald Knuth的一句话总结协程的特点：

**“子程序就是协程的一种特例。”**

## web框架中的进程和共享变量的问题

关键词： 绝大部分 python 框架都是多进程模型，或者叫 pre fork，或者叫 进程池

测试代码：[test-flask](./test-flask.py)