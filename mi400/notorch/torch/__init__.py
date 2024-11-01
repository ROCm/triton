from . import version
from . import cuda
from .device import device
from .types import float8_e5m2, float16, bfloat16, float32, float_ as float, float64, \
    int8, int16, int32, int_ as int, int64, uint8, uint16, uint32, uint64
from .tensor import Tensor, zeros, ones, empty, empty_like, abs_ as abs, \
    exp, max_ as max
from .tensor import Tensor as tensor
from .rand import manual_seed, rand, randn, randint
