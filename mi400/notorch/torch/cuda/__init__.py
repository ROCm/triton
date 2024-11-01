from .streams import Stream


def is_available():
    return True


def get_device_capability(device=None):
    return 9, 0


def current_device():
    return 0


def set_device(device):
    assert device == 0, "only a single GPU device is supported"


def current_stream(device):
    return Stream(device)
