import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm

class DRL(nn.Module):
    def __init__(self, args , apply_wn=False):
        super(DRL, self).__init__()
        
        source_num = len(args.evaluation_subjects)-1
        if apply_wn:
            self.DRL = weightNorm(self.DRL)
            
        self.DRL = ChannelAttention(source_num, args.reduction_ratio)             
        self.DRL.apply(init_weights)    

    def forward(self, x):
        x = self.DRL(x)
        x = torch.softmax(x, dim=-1)
        return x
    
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(
            self.n_channels_in / float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )

    def forward(self, x):

        kernel = x.size()[-1]
        avg_pool = F.avg_pool1d(x, kernel)
        max_pool = F.max_pool1d(x, kernel)

        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck
        
        sig_pool = torch.sigmoid(pool_sum)
        return sig_pool   

if __name__=='__main__':
    print()
    