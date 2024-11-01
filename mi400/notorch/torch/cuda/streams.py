from ..device import normalize_device


class Stream:

    def __init__(self, device):
        device = normalize_device(device)
        assert device.type == 'cuda'
        assert device.index == 0
        self.device = device
        self.cuda_stream = 0

    def __str__(self):
        return repr(self)

    def __repr__(self):
        cls_name = self.__class__.__module__ + '.' + self.__class__.__name__
        return f'<{cls_name} device={self.device} cuda_stream=0x{self.cuda_stream:x}>'
