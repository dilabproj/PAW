from torch.utils.data import Dataset

class dependent_loader(Dataset):
    def __init__(self, data, label):
        
        self.data,self.label=data, label
        
    def __len__(self):
        """'return the size of dataset"""
        return len(self.data)

    def __getitem__(self, index):
        data,label=self.data[index], self.label[index]

        return data, label

class loader_with_domain_label(Dataset):
    def __init__(self, data, label, domain_label):
        
        self.data,self.label=data, label
        self.domain_label = domain_label
        
    def __len__(self):
        """'return the size of dataset"""
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        return data, self.label[index],self.domain_label[index]