from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook

class RunOpAtEndHook(session_run_hook.SessionRunHook):

  def __init__(self, op_dict):
    """Ops is a dict of {name: op} to run before the session is destroyed."""
    self._ops = op_dict

  def end(self, session):
    for name in sorted(self._ops.keys()):
      logging.info('{0}: {1}'.format(name, session.run(self._ops[name])))