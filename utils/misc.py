import time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    def __init__(self):
        self.start = 0
        self.end = 0
        self.total = 0

    def tick(self):
        self.start = time.time()
        return self.start

    def toc(self):
        self.end = time.time()
        self.total = self.end - self.start
        return self.end

    def print_time(self, title):
        print(f'{title} time: {self.total:.4f}s')


def to_list(v):
    if not isinstance(v, list):
        return [v]
    return v