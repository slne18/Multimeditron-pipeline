from torch.utils.data import IterableDataset
from typing import List

class InterleavedDataset(IterableDataset):
    def __init__(self, datasets: List[IterableDataset]):
        super().__init__()
        self.current_ds_idx = 0
        self.datasets= datasets
        self.current_proc = 0
                
    def next_element(self, datasets):
        next_sample = next(datasets[self.current_ds_idx])        
        self.current_ds_idx = (self.current_ds_idx + 1) % len(datasets)
        self.current_proc += 1
        
        return next_sample
    
    def __iter__(self):        
        datasets = list(map(iter, self.datasets))
        self.current_ds_idx = 0
        
        while True:
            next_element = self.next_element(datasets)            
            yield next_element
            
