import tensorflow as tf

isTrue = lambda x: x == True
isFalse = lambda x: x == False
boolean = lambda x: isinstance(x, bool) or (isinstance(x, int) and (x == 0 or x == 1))

positiveFloat = lambda x: isinstance(x, float) and x > 0.0
positiveInt = lambda x: isinstance(x, int) and x > 0
existingStr = lambda x: isinstance(x, str)
percentage = lambda x: isinstance(x, float) and (0.0 <= x and x < 1.0)

tfVariable = lambda x: isinstance(x, tf.Variable)
tfTensor = lambda x: isinstance(x, tf.Tensor)
tfGraph = lambda x: isinstance(x, tf.Graph)

optionalTensor = lambda x: x is None or isinstance(x, tf.Tensor)
optionalStr = lambda x: x is None or isinstance(x, str)

floatOrTensor = lambda x: positiveFloat(x) or isinstance(x, tf.Tensor)

_checkList = lambda x, assrt: isinstance(x, list) and len([x_ for x_ in x if assrt(x_)]) == len(x)

nonEmptyList = lambda x: isinstance(x, list) and len(x) > 0
emptyList = lambda x: isinstance(x, list) and len(x) == 0

strList = lambda x: _checkList(x, lambda x_: isinstance(x_, str))
tupleList = lambda x: _checkList(x, lambda x_: isinstance(x_, tuple))
intList = lambda x: _checkList(x, lambda x_: isinstance(x_, int))
floatList = lambda x: _checkList(x, lambda x_: isinstance(x_, float))
tensorList = lambda x: _checkList(x, lambda x_: isinstance(x_, tf.Tensor))
variableList = lambda x: _checkList(x, lambda x_: isinstance(x_, tf.tf.Variable))
