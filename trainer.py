#!/usr/bin/python3

import tensorflow as tf
import os
import subprocess
import signal
from tf_modules.metrics.hardware_metrics import HardwareMetrics

class Trainer:

    def __init__(self, config, run_tensorboard=True, hardware=True, reset_log=False, port=6006):
        self.config = config

        if reset_log:
            try:
                self.config.clear_log()
            except:
                self.config.log_dir = "{}_1".format(self.config.log_dir)

        self.port = port
        self.current_epoch = 0
        self.run_tensorboard = run_tensorboard
        self.hardware = hardware

    def __enter__(self):
        tf.reset_default_graph()

        # We reset the configuration to ensure all tensorflow variables are within the current graph
        self.config.reset_tf_variables()

        if self.run_tensorboard:
            command = "tensorboard --logdir={} --port={}".format(os.path.abspath(self.config.log_dir), self.port)

            # Start tensorboard
            self.tensorboard = subprocess.Popen(command,
                                                stdout=subprocess.PIPE,
                                                shell=True,
                                                preexec_fn=os.setsid)
        return self

    def inspect_hardware(self):
        self.hardware_metrics = HardwareMetrics(self.config)

    def start_session(self, run_threads=False):
        if self.hardware:
            self.hardware_metrics = HardwareMetrics(self.config)

        self.session = tf.Session(graph=self.config.graph)
        self.session.run(tf.group(tf.global_variables_initializer(),
                                  tf.local_variables_initializer()))

        if run_threads:
            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(self.session, self.coord)

        return self.session

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(exc_type, exc_value, traceback)

        # Stop the queue runners once we are done running
        if hasattr(self, "coord"):
            self.coord.request_stop()
            self.coord.join(self.threads)

        # Stop tensorboard once we are done running
        if hasattr(self, "tensorboard"):
            try:
                os.killpg(os.getpgid(self.tensorboard.pid), signal.SIGTERM)
            except:
                pass

        # Close the session once we are done running
        if hasattr(self, "session"):
            self.session.close()

    def epochs(self):
        for i in range(self.config.epoch_amount):
            self.current_epoch += 1
            yield i

    def steps(self):
        for i in range(self.config.total_batches):
            if i > 0 and i % self.config.train_steps == 0:
                self.current_epoch += 1

            yield i

    def batches_pr_epoch(self):
        for i in range(self.config.train_steps):
            yield i

    def should_validate(self, step):
        if self.config.evaluate_at is None:
            return False

        validate = step % self.config.evaluate_at == 0

        # If it is time to validate we also meassure the hardware performance
        if hasattr(self, 'hardware_metrics') and validate:
            self.hardware_metrics.step(self.session)

        return validate

    def should_save(self, step):
        if self.config.save_at is None or self.config.save_at == 0:
            return False

        return step % self.config.save_at == 0

    def new_epoch(self, step):
        return step > 0 and step % self.config.train_steps == 0
