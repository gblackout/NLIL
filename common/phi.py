import torch
import torch.nn as nn
import torch.nn.functional as F
from common.cmd_args import cmd_args
from common import constants

class Phi(nn.Module):

    def __init__(self):
        super(Phi, self).__init__()

    def forward(self, noarg_input, arg_input=None):

        raise NotImplementedError


class Even(Phi):

    def forward(self, noarg_input, arg_input=None):

        return (1 - (noarg_input % 2)).type(torch.float)


class Succ(Phi):

    def forward(self, noarg_input, arg_input=None):
        # right shift by 1
        arg_input = torch.cat([torch.zeros(arg_input.size(0),
                                           arg_input.size(1), 1).to(cmd_args.device), arg_input], dim=-1)[:, :, :-1]
        return arg_input


class Zero(Phi):

    def forward(self, noarg_input, arg_input=None):

        return (noarg_input == 0).type(torch.float)


class Ident(Phi):

    def forward(self, noarg_input, arg_input=None):

        return arg_input


class TabularPhi(Phi):

    def __init__(self, name):
        super(TabularPhi, self).__init__()
        self.name = 'phi_' + name
        self.constmm = ConstMatMul.apply

    def forward(self, noarg_input, arg_input=None):
        """

        :param noarg_input:
            (b, num_const, 1) if unp, otherwise (num_const, num_const)
        :param arg_input:
            (b, num_arg, num_const, 1) or [(1, num_arg, num_const, num_const), (b, num_arg, num_const, 1)]
        :return:
            [(1, num_arg, num_const, num_const), (b, num_arg, num_const, 1)]
        """

        if arg_input is None:

            return noarg_input # (b, num_const, num_inst)

        else:
            # (1, num_const, num_const) X (b, num_const, num_inst) --> (b, num_const, num_inst)

            if self.name[:4] == constants.IDENT_PHI:
                return arg_input

            bsize, num_const = arg_input.size(0), arg_input.size(1)
            arg_input = arg_input.transpose(0, 1).view(num_const, -1)
            res = noarg_input.transpose(0, 1).mm(arg_input)
            res = res.view(num_const, bsize, -1).transpose(0, 1)

            return res

class ConstMatMul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x)
        with torch.no_grad():
            tmp = x.matmul(y)
        return tmp

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        z = torch.matmul(grad_output.transpose(1, 2), x).transpose(1, 2)
        return x, z