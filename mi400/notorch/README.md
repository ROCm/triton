# notorch
This is an extremely minimal shim for Pytorch. It implements the bare minimum
necessary for [Triton to work in FFM](https://amd.atlassian.net/wiki/spaces/MI400AGSSW/pages/588940366/Triton+on+FFM).
Its only dependencies are hip-python and Numpy.

This is adapted from Jungwook Park's work on Triton for MI350:
https://github.com/triton-lang/triton/compare/main...jungpark-mlir:triton:notorch

## Installation
The instructions below assume an ETX environment.

```shell
python3.12 -m venv venv
source venv/bin/activate.csh
pip install /proj/mi450_pemu_runs/users/merenber/ffm/hip-python/hip-python/dist/*.whl
pip install -e .
```

To create a wheel:
```shell
pip install build
python -m build
```
