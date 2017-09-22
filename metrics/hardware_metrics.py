import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import time
try:
    import psutil
    missing_psutil = False
except:
    missing_psutil = True

from extensions.utils import *
from extensions.assertions.checks import *
from collections import defaultdict
from subprocess import PIPE, Popen
import os

from threading import Thread
from collections import deque
from queue import Queue
import random

class GPU:
    def __init__(self, ID, temperature, load, memory, power):
        self.id = ID
        self.temperature = temperature
        self.gpu_load = load
        self.memory_load = memory
        self.power_usage = power

def getGPUs():
    # Get ID, processing and memory utilization for all GPUs
    try:
        p = Popen(["nvidia-smi","--query-gpu=index,temperature.gpu,utilization.gpu,utilization.memory,power.draw,", "--format=csv,noheader,nounits"], stdout=PIPE)
        output = p.stdout.read().decode('UTF-8')
        lines = output.split(os.linesep)
        numDevices = len(lines)-1
        GPUs = []
        for g in range(numDevices):
            line = lines[g]
            vals = line.split(', ')
            GPUs.append(GPU(ID=int(vals[0]),
                            temperature = float(vals[1]),
                            load=float(vals[2])/100,
                            memory=float(vals[3]),
                            power=float(vals[4])))
        return GPUs
    except:
        return []


scope = lambda x: 'GPU_{}'.format(x)

class HardwareMetrics(object):

    def __init__(self, config):
        self.config = config
        self.scope = 'hardware'
        self.collection = [self.scope]

        self.ops = defaultdict(dict)
        self.initialize_gpu_metrics()
        self.initialize_cpu_metrics()
        self.initialize_time_metrics()

        self.batch_times = []

        self.summaries = tf.summary.merge_all(self.scope)
        self.writer = tf.summary.FileWriter('{}/{}'.format(self.config.log_dir, self.scope))

        self.stop_queues = []
        self.monitors = []
        self.gpu_queue = self.start_gpu_monitoring()
        if not missing_psutil:
            self.cpu_queue = self.start_cpu_monitoring()


    def start_gpu_monitoring(self):
        def monitor(out_dequeue, stop_queue):
            while True:
                try:
                    stop_queue.get_nowait()
                    break
                except:
                    pass

                out_dequeue.append(getGPUs())

                while len(out_dequeue) > 100:
                    out_dequeue.popleft()

        results = deque()
        stop_queue = Queue()

        m = Thread(target = monitor, args = (results, stop_queue))
        m.start()

        self.monitors.append(m)
        self.stop_queues.append(stop_queue)

        return results

    def start_cpu_monitoring(self):
        def monitor(out_dequeue, stop_queue):
            while True:
                try:
                    stop_queue.get_nowait()
                    break
                except:
                    pass

                cpu = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory().percent
                out_dequeue.append((cpu, memory))

                while len(out_dequeue) > 100:
                    out_dequeue.popleft()

        results = deque()
        stop_queue = Queue()

        for i in range(3):
            m = Thread(target = monitor, args = (results, stop_queue))
            m.start()
            self.monitors.append(m)
            time.sleep(random.random())

        self.stop_queues.append(stop_queue)

        return results


    def initialize_gpu_metrics(self):
        for gpu in getGPUs():
            _id = scope(gpu.id)
            with tf.name_scope('X-Metrics---{}'.format(_id)):
                temperature = tf.Variable(0, dtype=tf.float32, trainable=False)
                gpu_load = tf.Variable(0, dtype=tf.float32, trainable=False)
                memory_load = tf.Variable(0, dtype=tf.float32, trainable=False)
                power_usage = tf.Variable(0, dtype=tf.float32, trainable=False)

                tf.summary.scalar('Temperature', temperature, collections=self.collection)
                tf.summary.scalar('Utilization', gpu_load, collections=self.collection)
                tf.summary.scalar('Memory_Utilization', memory_load, collections=self.collection)
                tf.summary.scalar('Power_Usage', power_usage, collections=self.collection)

                self.ops[_id]['temperature'] = temperature
                self.ops[_id]['gpu_load'] = gpu_load
                self.ops[_id]['memory_load'] = memory_load
                self.ops[_id]['power_usage'] = power_usage


    def initialize_cpu_metrics(self):
        with tf.name_scope('X-Metrics---CPU'):
            cpu_usage = tf.Variable(0, dtype=tf.float32, trainable=False)
            memory_usage = tf.Variable(0, dtype=tf.float32, trainable=False)
            tf.summary.scalar('CPU_Utilization', cpu_usage, collections=self.collection)
            tf.summary.scalar('Memory_Utilization', memory_usage, collections=self.collection)

            self.ops['cpu']['cpu_usage'] = cpu_usage
            self.ops['cpu']['memory_usage'] = memory_usage

    def add_batch_time(self, batch_time):
        self.batch_times.append(batch_time)

    def initialize_time_metrics(self):
        with tf.name_scope('X-Metrics---TIME'):
            batch_time = tf.Variable(0, dtype=tf.float32, trainable=False)
            time_left = tf.Variable(0, dtype=tf.float32, trainable=False)
            tf.summary.scalar('Seconds_Pr_Batch', batch_time, collections=self.collection)
            tf.summary.scalar('Hours_Left', time_left, collections=self.collection)

            self.ops['time']['batch'] = batch_time
            self.ops['time']['left'] = time_left


    def step(self, sess, step):
        feed_dict = {}

        if self.batch_times:
            mean = np.array(self.batch_times).mean()
            time_left = (self.config.total_batches - step) * mean

            feed_dict[self.ops['time']['batch']] = mean
            feed_dict[self.ops['time']['left']] = time_left / 60 / 60
            self.batch_times = []

        try:
            for gpu in self.gpu_queue.pop():
                _id = scope(gpu.id)
                feed_dict.update({self.ops[_id]['temperature']: gpu.temperature,
                                  self.ops[_id]['gpu_load']:    gpu.gpu_load,
                                  self.ops[_id]['memory_load']: gpu.memory_load,
                                  self.ops[_id]['power_usage']: gpu.power_usage})
        except:
            pass

        try:
            cpu, memory = self.cpu_queue.pop()

            feed_dict[self.ops['cpu']['cpu_usage']]    = cpu
            feed_dict[self.ops['cpu']['memory_usage']] = memory
        except:
            pass

        if feed_dict:
            summaries = sess.run(self.summaries, feed_dict=feed_dict)
            self.writer.add_summary(summaries, step)
            self.writer.flush()
