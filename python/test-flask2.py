from flask import Flask
import time, threading, multiprocessing, os

app = Flask(__name__)

sample_queue = multiprocessing.Queue()
data = []

cnt = 0
@app.route('/')
def index():
    global cnt, data
    cnt += 1
    sample_queue.put(cnt)
    data = data + [cnt]
    return str(os.getpid())+'\t'+str(cnt)+'\t'+str(data)

def f1():
    while True:
        t = sample_queue.get()
        print(t)
        print('f1:',os.getpid(), os.getppid(),len(data))
        time.sleep(1)

if __name__ == '__main__':
    print(os.getpid(), os.getppid(),len(data))
    t = multiprocessing.Process(target=f1)
    t.start()

    app.run(debug=True, host='0.0.0.0')