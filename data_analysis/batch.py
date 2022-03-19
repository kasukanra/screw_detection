import numpy as np

class Batch:
    def __init__(self, num_items, batch_size, seed=0):
        self.indices = np.arange(num_items)
        self.num_items = num_items
        self.batch_size = batch_size
        self.rnd = np.random.RandomState(seed)
        self.rnd.shuffle(self.indices)
        self.ptr = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.ptr + self.batch_size > self.num_items:
            self.rnd.shuffle(self.indices)
            self.ptr = 0
            raise StopIteration  
        else:
            result = self.indices[self.ptr:self.ptr+self.batch_size]
            self.ptr += self.batch_size
            return result