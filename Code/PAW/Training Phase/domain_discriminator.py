import torch.nn as nn

class Discriminator(nn.Module):
    """
        Simple Discriminator w/ MLP
    """
    def __init__(self, input_size=512, num_classes=1):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, int(input_size/2)),
            nn.LeakyReLU(0.2),
            nn.Linear( int(input_size/2),  int(input_size/4)),
            nn.LeakyReLU(0.2),
            nn.Linear( int(input_size/4), num_classes),
            nn.Sigmoid(),
        )
    
    def forward(self, h):
        y = self.layer(h)
        return y