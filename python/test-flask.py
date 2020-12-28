from flask import Flask
import time, threading
import gevent, os
app = Flask(__name__)

data = []

cnt = 0
@app.route('/')
def index():
    global cnt, data
    cnt += 1
    data = data + [cnt]
    return str(os.getpid())+'\t'+str(cnt)+'\t'+str(data)

def f1():
    while True:
        print('f1:',os.getpid(), os.getppid(),len(data))
        time.sleep(1)

if __name__ == '__main__':
    print(os.getpid(), os.getppid(),len(data))
    t = threading.Thread(target=f1)
    t.start()

    app.run(debug=True, host='0.0.0.0')