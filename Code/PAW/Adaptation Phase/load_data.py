from torch.utils.data import Dataset
from augmentation import *

class my_loader(Dataset):
    def __init__(self, data, label, if_test=False):
        
        self.data,self.label=data, label
        self.aug=weak_aug()
        self.if_test=if_test
        
    def __len__(self):
        """'return the size of dataset"""
        return len(self.data)

    def __getitem__(self, index):
        
        data,label=self.data[index], self.label[index]
        if not self.if_test:
            data=self.aug(data)

        return data, label, index
