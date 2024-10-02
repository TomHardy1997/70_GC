# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class NLLSurvLoss(nn.Module):
#     """
#     The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
#     Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
#     Parameters
#     ----------
#     alpha: float
#         Weighting factor for the uncensored loss term.
#     eps: float
#         Numerical constant; lower bound to avoid taking logs of tiny numbers.
#     reduction: str
#         Reduction method: 'mean' or 'sum'.
#     """
#     def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
#         super().__init__()
#         self.alpha = alpha
#         self.eps = eps
#         self.reduction = reduction

#     def forward(self, h, y, t, c):
#         """
#         Parameters
#         ----------
#         h: (n_batches, n_classes)
#             The neural network output discrete survival predictions such that hazards = sigmoid(h).
#         y: (n_batches, 1)
#             The true time bin label (first column).
#         t: (n_batches, 1)
#             The true event times.
#         c: (n_batches, 1)
#             The censorship indicator (second column).
#         """
#         return nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
#                         alpha=self.alpha, eps=self.eps,
#                         reduction=self.reduction)

# def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='mean'):
#     """
#     The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
#     Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
#     Parameters
#     ----------
#     h: (n_batches, n_classes)
#         The neural network output discrete survival predictions such that hazards = sigmoid(h).
#     y: (n_batches, 1)
#         The true time bin index label.
#     c: (n_batches, 1)
#         The censoring status indicator.
#     alpha: float
#         Weighting factor for the uncensored loss term.
#     eps: float
#         Numerical constant; lower bound to avoid taking logs of tiny numbers.
#     reduction: str
#         Reduction method: 'mean' or 'sum'.
#     References
#     ----------
#     Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
#     """
#     y = y.type(torch.int64)
#     c = c.type(torch.int64)

#     hazards = torch.sigmoid(h)
#     S = torch.cumprod(1 - hazards, dim=1)

#     S_padded = torch.cat([torch.ones_like(c), S], 1)
#     s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
#     h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
#     s_this = torch.gather(S_padded, dim=1, index=y + 1).clamp(min=eps)

#     uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
#     censored_loss = - c * torch.log(s_this)

#     neg_l = censored_loss + uncensored_loss
#     if alpha is not None:
#         loss = (1 - alpha) * neg_l + alpha * uncensored_loss

#     if reduction == 'mean':
#         loss = loss.mean()
#     elif reduction == 'sum':
#         loss = loss.sum()
#     else:
#         raise ValueError(f"Bad input for reduction: {reduction}")

#     return loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def forward(self, h, y, t, c):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y: (n_batches, 1)
            The true time bin label (first column).
        t: (n_batches, 1)
            The true event times.
        c: (n_batches, 1)
            The censorship indicator (second column).
        """
        # 确保所有张量都在相同设备上
        device = h.device  # 获取h所在设备
        y, t, c = y.to(device), t.to(device), c.to(device)
        
        return nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)


def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='mean'):
    """
    The negative log-likelihood loss function for the discrete time to event model.
    """
    # 确保y和c在同一设备上
    y = y.type(torch.int64).to(h.device)
    c = c.type(torch.int64).to(h.device)

    hazards = torch.sigmoid(h)
    S = torch.cumprod(1 - hazards, dim=1)

    # 确保S_padded在同一设备上
    S_padded = torch.cat([torch.ones_like(c, device=h.device), S], 1)
    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y + 1).clamp(min=eps)

    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError(f"Bad input for reduction: {reduction}")

    return loss


class coxph_loss(torch.nn.Module):

    def __init__(self):
        super(coxph_loss, self).__init__()

    def forward(self, risk, phase, censors):
        
        sort_idx = torch.argsort(phase, descending=True)
        risk = risk[sort_idx]
        phase = phase[sort_idx]
        censors = censors[sort_idx]
        #riskmax = risk
        riskmax = F.normalize(risk, p=2, dim=0)

        log_risk = torch.log((torch.cumsum(torch.exp(riskmax), dim=0)))

        uncensored_likelihood = torch.add(riskmax, -log_risk)
        resize_censors = censors.resize_(uncensored_likelihood.size()[0], 1)
        censored_likelihood = torch.mul(uncensored_likelihood, resize_censors)

        loss = -torch.sum(censored_likelihood) / float(censors.nonzero().size(0))
        #loss = -torch.sum(censored_likelihood) / float(censors.size(0))

        return loss