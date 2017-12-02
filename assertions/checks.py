import tensorflow as tf

isTrue          = lambda x: True if x else False
isFalse         = lambda x: not isTrue(x)
boolean         = lambda x: isinstance(x, bool) or (isinstance(x, int) and (x == 0 or x == 1))

positiveFloat   = lambda x: isinstance(x, float) and x > 0.0
positiveInt     = lambda x: isinstance(x, int) and x > 0
existingStr     = lambda x: isinstance(x, str)
percentage      = lambda x: isinstance(x, float) and (0.0 <= x < 1.0)

tfVariable      = lambda x: isinstance(x, tf.Variable)
tfTensor        = lambda x: isinstance(x, tf.Tensor)
tfGraph         = lambda x: isinstance(x, tf.Graph)

optionalTensor  = lambda x: x is None or isinstance(x, tf.Tensor)
tensorOrBool    = lambda x: tfTensor(x) or boolean(x)
optionalStr     = lambda x: x is None or isinstance(x, str)

floatOrTensor   = lambda x: positiveFloat(x) or isinstance(x, tf.Tensor)

nonEmptyList    = lambda x: isinstance(x, list) and len(x) > 0
emptyList       = lambda x: isinstance(x, list) and len(x) == 0

_checkList      = lambda assrt, x: isinstance(x, list) and False not in [assrt(z) for z in x]
strList         = lambda x: _checkList(lambda x_: isinstance(x_, str),         x)
tupleList       = lambda x: _checkList(lambda x_: isinstance(x_, tuple),       x)
intList         = lambda x: _checkList(lambda x_: isinstance(x_, int),         x)
floatList       = lambda x: _checkList(lambda x_: isinstance(x_, float),       x)
tensorList      = lambda x: _checkList(lambda x_: isinstance(x_, tf.Tensor),   x)
variableList    = lambda x: _checkList(lambda x_: isinstance(x_, tf.Variable), x)
