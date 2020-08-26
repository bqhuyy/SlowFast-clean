import torch
import torch.nn as nn
import torch.nn.functional as F

# @torch.jit.script
# def shift_ops(x, fold):
#     out = torch.zeros_like(x)
#     out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
#     out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
#     out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

#     return out

@torch.jit.script
def shift_ops1(input, buffer, fold):
    buffer[:, :-1] = input.data[:, 1:, :fold]
    return buffer

@torch.jit.script
def shift_ops2(input, buffer, fold):
    input.data[:, :, :fold] = buffer
    return input

@torch.jit.script
def shift_ops3(input, buffer, fold):
    buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
    return buffer

@torch.jit.script
def shift_ops4(input, buffer, fold):
    input.data[:, :, fold: 2 * fold] = buffer
    return input


class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):
        x = self.shift(x, n_segment=self.n_segment, fold_div=self.fold_div)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            fold = c // torch.tensor(fold_div)
            out = InplaceShift.apply(x, fold)
        else:
            fold = c // fold_div
            
            out = torch.zeros_like(x)
            out[:, :-1, : 2*fold] = x[:, 1:, : 2*fold]  # shift left
            # out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            # out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)
    
    
class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new_zeros(n, t, fold, h, w)
        buffer = shift_ops1(input, buffer, fold)
        input = shift_ops2(input, buffer, fold)
        buffer.zero_()
        buffer = shift_ops3(input, buffer, fold)
        input = shift_ops4(input, buffer, fold)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None