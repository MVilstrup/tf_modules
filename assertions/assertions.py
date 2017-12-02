import inspect
from tf_modules.assertions.checks import (tupleList)


class AssertionClass(object):
    def __init__(self):
        self._class_name = type(self).__name__

    def _print(self, _assert):
        return inspect.getsource(_assert).strip()

    def assertions(self, _assertions):
        msg = "Assertions should either be list of tuples or dict"
        assert isinstance(_assertions, dict) or isinstance(_assertions, list), msg

        tests = []
        if isinstance(_assertions, dict):
            for attr, _assert in _assertions.items():
                self.assertion((attr, _assert))
        else:
            assert (tupleList(_assertions)), msg
            map(self.assertion, _assertions)

    def assertion(self, _assertion):

        msg = "assertion should be a tuple of attribute and function"
        assert isinstance(_assertion, tuple), msg

        attr, _assert = _assertion

        msg = "{} should include a {}".format(self._class_name, attr)
        assert hasattr(self, attr), msg

        msg = "{} has to live up to {}".format(attr, self._print(_assert))
        assert _assert(getattr(self, attr)), msg

    def _check(self, _assertion):

        assert isinstance(_assertion, tuple), "Condition should be a tuple"
        attr, _assert = _assertion

        msg = "{} should include a {}".format(self._class_name, attr)
        assert hasattr(self, attr), msg
        return _assert(getattr(self, attr))

    def condition(self, cond, assertion):

        if self._check(cond):
            self.assertion(assertion)

    def either(self, _assrt1, _assrt2):

        checked_1 = self._check(_assrt1)
        attr1, _assert1 = _assrt1

        checked_2 = self._check(_assrt2)
        attr2, _assert2 = _assrt2

        msg = "Either: \n{} ({}) \nor \n{} ({}) should be correct".format(attr1,
                                                                          self._print(_assert1),
                                                                          attr2,
                                                                          self._print(_assert2))
        assert checked_1 or checked_2, msg

    def both(self, _assrt1, _assrt2):

        checked_1 = self._check(_assrt1)
        attr1, _assert1 = _assrt1

        checked_2 = self._check(_assrt2)
        attr2, _assert2 = _assrt2

        msg = "Both: \n{} ({}) \nand \n{} ({}) should be correct".format(attr1,
                                                                         self._print(_assert1),
                                                                         attr2,
                                                                         self._print(_assert2))
        assert checked_1 and checked_2, msg
