class device:

    def __init__(self, type_, index=None):
        if index is not None:
            self.type = type_
            self.index = index
        elif isinstance(type_, int):
            self.type = 'cuda'
            self.index = type_
        elif ':' in type_:
            self.type = type_.split(':')[0]
            self.index = int(type_.split(':')[1])
        else:
            self.type = type_
            self.index = 0
        assert self.type in ['cuda', 'cpu']
        assert self.index == 0

    def __str__(self):
        if self.type == 'cpu':
            return 'cpu'
        return f'{self.type}:{self.index}'

    def __repr__(self):
        if self.type == 'cpu':
            return "device(type='cpu')"
        return f"device(type='{self.type}', index={self.index})"

    def __eq__(self, other):
        if isinstance(other, device):
            return self.type == other.type and self.index == other.index
        elif isinstance(other, int):
            return self.type == 'cuda' and self.index == other
        elif isinstance(other, str):
            return self.type == other
        return False


def normalize_device(dev):
    if dev is None:
        return device('cpu')
    if isinstance(dev, device):
        return dev
    return device(dev)
