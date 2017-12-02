from tensorflow.python.training import session_run_hook
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util


class EarlyStoppingHook(session_run_hook.SessionRunHook):
    """Monitor to request stop when loss stops decreasing."""

    def __init__(self,
                 early_stopping_rounds,
                 early_stopping_loss_threshold=None,
                 loss_op=None):
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_loss_threshold = early_stopping_loss_threshold
        self.loss_op = loss_op
        self.min_loss = None
        self.last_step = -1
        # self.steps records the number of steps for which the loss has been
        # non-decreasing
        self.steps = 0

    def before_run(self, run_context):
        loss = (self.loss_op if self.loss_op is not None else
                run_context.session.graph.get_operation_by_name(
                    LOSS_NAME).outputs[0])
        return session_run_hook.SessionRunArgs(
            {'global_step': training_util.get_global_step(),
             'current_loss': loss})

    def after_run(self, run_context, run_values):
        current_loss = run_values.results['current_loss']
        current_step = run_values.results['global_step']
        self.steps += 1
        # Guard against the global step going backwards, which might happen
        # if we recover from something.
        if self.last_step == -1 or self.last_step > current_step:
            logging.info('TensorForestLossHook resetting last_step.')
            self.last_step = current_step
            self.steps = 0
            self.min_loss = None
            return

        self.last_step = current_step
        if (self.min_loss is None or current_loss <
            (self.min_loss - self.min_loss * self.early_stopping_loss_threshold)):
            self.min_loss = current_loss
            self.steps = 0
        if self.steps > self.early_stopping_rounds:
            logging.info('TensorForestLossHook requesting stop.')
            run_context.request_stop()
