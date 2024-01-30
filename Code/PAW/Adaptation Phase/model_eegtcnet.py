import torch
from torch import nn
from torch.nn.functional import elu
from torch.nn import init
from torch.nn.utils import weight_norm


class EEGTCNet(nn.Module):
    def __init__(self,nb_classes,Chans=64, Samples=128, layers=3, kernel_s=10,
                 filt=10, dropout=0, activation='relu', F1=4, D=2, kernLength=64, dropout_eeg=0.1):
        super().__init__()
        
        numFilters = F1
        F2= numFilters*D
        self.EEGNet=EEGNet(F1=F1,kernel_length=kernLength,D=D,in_chans=Chans,drop_prob=dropout_eeg)
        self.TCN_block=TCN(n_in_chans=F2,n_blocks=layers,kernel_size=kernel_s,n_filters=filt,drop_prob=dropout,activation=activation)
        #self.feature=nn.Sequential(EEGNet(F1=F1,kernel_length=kernLength,D=D,in_chans=Chans,drop_prob=dropout_eeg),
        #                           TCN(n_in_chans=F2,n_blocks=layers,kernel_size=kernel_s,n_filters=filt,drop_prob=dropout,activation=activation))
        self.dense=nn.Linear(filt, nb_classes)
        
    def forward(self,x):
        x=self.EEGNet(x)
        feature=self.TCN_block(x)
        #feature = self.feature(x)
        x=self.dense(feature)
        return x#,feature


class EEGNet(nn.Module):
    def __init__(
        self,
        in_chans,
        input_window_samples=None,
        pool_mode="mean",
        F1=8,
        D=2,
        F2=16,  
        kernel_length=64,
        third_kernel_size=(8, 4),
        drop_prob=0.25,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.input_window_samples = input_window_samples
        self.pool_mode = pool_mode
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.third_kernel_size = third_kernel_size
        self.drop_prob = drop_prob

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        self.ensuredims = Ensure4d()
        # b c 0 1
        # now to b 1 0 c
        self.dimshuffle = Expression(_transpose_to_b_1_c_0)

        self.conv_temporal=nn.Conv2d(1, self.F1, (1, self.kernel_length), stride=1, bias=False, padding='same')
        
        self.bnorm_temporal=nn.BatchNorm2d(self.F1, momentum=0.1, affine=True)
        
        # Block 2
        self.conv_spatial=nn.Conv2d(self.F1, self.F1 * self.D, (self.in_chans, 1), stride=1, bias=False, groups=self.F1, padding=(0, 0))
        self.bnorm_1=nn.BatchNorm2d(self.F1 * self.D, momentum=0.1, affine=True)#Conv2dWithConstraint
        self.elu_1=Expression(elu)
        self.pool_1=pool_class(kernel_size=(1, 8))
        self.drop_1=nn.Dropout(p=self.drop_prob)

        # block 3
        self.conv_separable_depth=nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16), stride=1, 
                                            bias=False, groups=self.F1 * self.D, padding='same')
        self.conv_separable_point=nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), stride=1, 
                                            bias=False, padding=(0, 0))

        self.bnorm_2 = nn.BatchNorm2d(self.F2, momentum=0.1, affine=True)
        self.elu_2 = Expression(elu)
        self.pool_2 = pool_class(kernel_size=(1, 8), stride=(1, 8))
        self.drop_2 = nn.Dropout(p=self.drop_prob)

        _glorot_weight_zero_bias(self)

    def forward(self, x):
        # layer 1
        x=self.ensuredims(x)
        x=self.dimshuffle(x)
        x=self.conv_temporal(x)
        x=self.bnorm_temporal(x)

        # layer 2
        x=self.conv_spatial(x)
        x=self.bnorm_1(x)
        x=self.elu_1(x)
        x=self.pool_1(x)
        x=self.drop_1(x)

        # layer 3
        x=self.conv_separable_depth(x)
        x=self.conv_separable_point(x)
        x=self.bnorm_2(x)
        x=self.elu_2(x)
        x=self.pool_2(x)
        x=self.drop_2(x)
        x = x.squeeze(2)
        return x

class TCN(nn.Module):

    def __init__(self, n_in_chans,n_blocks, n_filters, kernel_size,
                 drop_prob, activation='relu'):
        super().__init__()
        activation_mode = dict(relu=nn.ReLU(), elu=nn.ELU())[activation]
        #self.ensuredims = Ensure4d()
        t_blocks = nn.Sequential()
        for i in range(n_blocks):
            n_inputs = n_in_chans if i == 0 else n_filters
            dilation_size = 2 ** i
            t_blocks.add_module("temporal_block_{:d}".format(i), TemporalBlock(
                n_inputs=n_inputs,
                n_outputs=n_filters,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size,
                drop_prob=drop_prob,
                activation=activation_mode
            ))
        self.temporal_blocks = t_blocks

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """

        x = self.temporal_blocks(x)

        x = x[:,:,-1]
        
        return x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 padding, drop_prob, activation):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.batch1=nn.BatchNorm1d(12, momentum=0.1, affine=True)
        self.chomp1 = Chomp1d((kernel_size - 1) * dilation)
        self.act1 = activation#nn.ReLU()
        self.dropout1 = nn.Dropout2d(drop_prob)

        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.batch2=nn.BatchNorm1d(12, momentum=0.1, affine=True)
        self.chomp2 = Chomp1d((kernel_size - 1) * dilation)
        self.act2 = activation
        self.dropout2 = nn.Dropout2d(drop_prob)

        self.downsample = (nn.Conv1d(n_inputs, n_outputs, 1)
                           if n_inputs != n_outputs else None)
        self.act = activation

        init.normal_(self.conv1.weight, 0, 0.01)
        init.normal_(self.conv2.weight, 0, 0.01)
        if self.downsample is not None:
            init.normal_(self.downsample.weight, 0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.batch1(out)
        out = self.act1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.batch2(out)
        out = self.act2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.act(out + res)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def extra_repr(self):
        return 'chomp_size={}'.format(self.chomp_size)

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)
    
class Ensure4d(nn.Module):
    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        return x


class Expression(nn.Module):
    """Compute given expression on forward pass.

    Parameters
    ----------
    expression_fn : callable
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (
            self.__class__.__name__ +
            "(expression=%s) " % expression_str
        )

def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)

def _glorot_weight_zero_bias(model):
    """Initalize parameters of all modules by initializing weights with
    glorot
     uniform/xavier initialization, and setting biases to zero. Weights from
     batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        if hasattr(module, "weight"):
            if not ("BatchNorm" in module.__class__.__name__):
                #nn.init.xavier_uniform_(module.weight, gain=1)
                #init.xavier_normal_(module.weight)
                init.normal_(module.weight, mean=0, std=0.1)
            else:
                nn.init.constant_(module.weight, 1)            
            
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
