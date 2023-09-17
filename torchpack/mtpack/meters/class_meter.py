from .meter import Meter

__all__ = ['TopKClassMeter']


class TopKClassMeter(Meter):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def reset(self):
        self.num_examples = 0
        self.num_correct = 0

    def update(self, outputs, targets):
        _, indices = outputs.topk(self.k, 1, True, True)

        indices = indices.transpose(0, 1)
        masks = indices.eq(targets.view(1, -1).expand_as(indices))

        self.num_examples += targets.size(0)
        print("self.num_correct: ", self.num_correct) 
        print("masks tanform shape: ", masks[:self.k].view(-1).float().sum(0).shape) 
        self.num_correct += masks[:self.k].view(-1).float().sum(0)

    def compute(self):
        return self.num_correct / max(self.num_examples, 1) * 100.

    def data(self):
        return {'num_examples': self.num_examples,
                'num_correct': self.num_correct}
    
    def set(self, data):
        if 'num_examples' in data:
            self.num_examples = data['num_examples']
        if 'num_correct' in data:
            self.num_correct = data['num_correct']
