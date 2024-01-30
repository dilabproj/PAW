import torch
from torch import nn
from torch.nn.functional import elu

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    def __init__(
        self,
        in_chans,
        n_classes,
        input_window_samples,
        pool_mode="mean",
        F1=8,
        D=2,
        F2=16,  # usually set to F1*D (?)
        kernel_length=64,
        third_kernel_size=(8, 4),
        drop_prob=0.25,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.pool_mode = pool_mode
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.third_kernel_size = third_kernel_size
        self.drop_prob = drop_prob

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]


        
        # feature extractor
        self.block1=nn.Sequential()
        self.block1.add_module("conv_temporal",nn.Conv2d(1,self.F1,(1, self.kernel_length),stride=1,bias=False,padding='same'))
        self.block1.add_module("bnorm_temporal",nn.BatchNorm2d(self.F1, momentum=0.1, affine=True))
        self.block1.add_module("conv_spatial",Conv2dWithConstraint(self.F1, self.F1 * self.D,(self.in_chans, 1),max_norm=1,stride=1,bias=False,groups=self.F1,padding='valid'))
        self.block1.add_module("bnorm_1",nn.BatchNorm2d(self.F1 * self.D, momentum=0.1, affine=True))
        self.block1.add_module("elu_1",Expression(elu))
        self.block1.add_module("pool_1",pool_class(kernel_size=(1, 4), stride=(1, 4)))
        self.block1.add_module("drop_1",nn.Dropout(p=self.drop_prob))
        

        self.block2=nn.Sequential()
        self.block2.add_module("conv_separable_depth",nn.Conv2d(self.F1 * self.D,self.F1 * self.D,(1, 16),stride=1,bias=False,groups=self.F1 * self.D, padding='same'))
        self.block2.add_module("conv_separable_point",nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), stride=1, bias=False, padding='same'))
        self.block2.add_module("bnorm_2",nn.BatchNorm2d(self.F2, momentum=0.1, affine=True))
        self.block2.add_module("elu_2",Expression(elu))
        self.block2.add_module("pool_2",pool_class(kernel_size=(1, 8), stride=(1, 8)))
        self.block2.add_module("drop_2",nn.Dropout(p=self.drop_prob))

        self.feature=nn.Sequential()
        self.feature.add_module('preprocess',nn.Sequential(Ensure4d(),Expression(_transpose_to_b_1_c_0)))
        self.feature.add_module("block1", self.block1)
        self.feature.add_module("block2", self.block2)
        self.feature.add_module('flatten', nn.Flatten())

        # class classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('dense', nn.Sequential(nn.Linear(F2*int(input_window_samples/32), n_classes*F2*int(input_window_samples/32)),
                                        nn.Linear(n_classes*F2*int(input_window_samples/32), n_classes)))

        _glorot_weight_zero_bias(self)

    def forward(self,x):
        
        feature=self.feature(x)
        out=self.class_classifier(feature)
        
        return out, feature


def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)


def _transpose_1_0(x):
    return x.permute(0, 1, 3, 2)



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
                nn.init.normal_(module.weight, mean=0, std=0.1)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


# %% 下方偏向輔助性質的function

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
def squeeze_final_output(x):
    """Removes empty dimension at end and potentially removes empty time
     dimension. It does  not just use squeeze as we never want to remove
     first dimension.

    Returns
    -------
    x: torch.Tensor
        squeezed tensor
    """

    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x
