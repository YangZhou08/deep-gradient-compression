from abc import abstractmethod

__all__ = ['Meter']


class Meter:
    def __init__(self):
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, *inputs):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def data(self):
        pass

    @abstractmethod
    def set(self, data):
        pass
