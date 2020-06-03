import torch
import math
from torch.optim.optimizer import Optimizer


class NewOptimizer(Optimizer):

    def __init__(self, params, lr=0.01, p_bound=None):
        """
        Arguments:
            lr (float): The learning rate. 0.01 is a good initial value to try.
            p_bound (float): Restricts the optimisation to a bounded set. A
                value of 2.0 restricts parameter norms to lie within 2x their
                initial norms. This regularises the model class.
        """
        self.p_bound = p_bound
        defaults = dict(lr=lr)
        super(NewOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0 and self.p_bound is not None:
                    state['max'] = self.p_bound * p.norm().item()

                d_p = p.grad.data

                # frob_norm_term = p.data.norm() ** 2
                # frob_deriv = 2*p.data

                U, S, V = torch.svd(p.data)
                # sing_val_term = S[0]
                sing_val_derivative = U[0].view(U.shape[0], 1) @ V[:, 0].view(1, V.shape[0])
                #
                # reg_term = (sing_val_derivative * frob_norm_term - frob_deriv * sing_val_term)/(frob_norm_term * frob_norm_term)

                reg_term = sing_val_derivative
                final_sub_term = d_p + reg_term

                normalization_term = final_sub_term.norm()
                p_norm = p.norm()

                if p_norm > 0.0 and normalization_term > 0.0:
                    p.data.add_(-group['lr'], final_sub_term * (1.0/normalization_term))
                else:
                    p.data.add_(-group['lr'], d_p)
                # p.data /= math.sqrt(1 + group['lr'] ** 2)

                if self.p_bound is not None:
                    p_norm = p.norm().item()
                    if p_norm > state['max']:
                        p.data *= state['max'] / p_norm

        return loss