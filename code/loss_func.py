import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedSurvLoss(nn.Module):
    """
    Combined survival loss function:
    - NLLSurvLoss (negative log-likelihood loss for survival analysis)
    - Rank Loss (for ranking risk scores)
    """
    def __init__(self, alpha=0.0, lambda_rank=0.5, eps=1e-7, reduction='mean'):
        super(CombinedSurvLoss, self).__init__()
        self.alpha = alpha
        self.lambda_rank = lambda_rank
        self.eps = eps
        self.reduction = reduction

    def forward(self, outputs, y, t, c):  # 修改 h -> outputs
        device = outputs.device
        y, t, c = y.to(device), t.to(device), c.to(device)
        # import ipdb;ipdb.set_trace()
        # Loss 1: NLLSurvLoss
        loss_nll = self.nll_loss(h=outputs, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                                 alpha=self.alpha, eps=self.eps, reduction=self.reduction)

        # 根据 censor = 1 表示删失，censor = 0 表示事件发生，生成 fc_mask2
        fc_mask2 = (c == 0).float()  # 1 表示事件发生，0 表示删失

        # Loss 2: Rank Loss
        loss_rank = self.rank_loss(outputs=outputs, t=t, fc_mask2=fc_mask2)  # 修改 h -> outputs

        # Combine the losses
        combined_loss = loss_nll + self.lambda_rank * loss_rank
        # print(f"NLL Loss: {loss_nll.item()}, Rank Loss: {loss_rank.item()}, Total Loss: {combined_loss.item()}")
        return combined_loss

    def nll_loss(self, h, y, c, alpha=0.0, eps=1e-7, reduction='mean'):
        """
        NLLSurvLoss: The negative log-likelihood loss function for survival analysis.
        """
        y = y.type(torch.int64).to(h.device)
        c = c.type(torch.int64).to(h.device)

        hazards = torch.sigmoid(h)
        S = torch.cumprod(1 - hazards, dim=1)

        S_padded = torch.cat([torch.ones_like(c, device=h.device), S], 1)
        s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
        h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
        s_this = torch.gather(S_padded, dim=1, index=y + 1).clamp(min=eps)

        uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
        censored_loss = - c * torch.log(s_this)

        total_loss = censored_loss + uncensored_loss
        if alpha is not None:
            total_loss = (1 - alpha) * total_loss + alpha * uncensored_loss

        if reduction == 'mean':
            return total_loss.mean()
        elif reduction == 'sum':
            return total_loss.sum()
        else:
            raise ValueError(f"Unsupported reduction type: {reduction}")
        
    # def rank_loss(self, outputs, t, fc_mask2):
    #     """
    #     Rank Loss for comparing risk scores in survival analysis.
    #     - outputs: logits from the model (before converting to risk scores)
    #     - t: survival time
    #     - fc_mask2: mask for censored samples (0 for censored, 1 for event/death)
    #     """
    #     one_vector = torch.ones_like(t, dtype=torch.float32).unsqueeze(1)  # 确保 one_vector 形状为 (batch_size, 1)
        
    #     # Step 1: Convert logits to risk scores
    #     risk_scores = -torch.sum(torch.cumprod(1 - torch.sigmoid(outputs), dim=1), dim=1)  # shape: (batch_size,)
    #     print(f"Risk Scores: {risk_scores}")
    #     # print(f"Risk Score Differences: {risk_diff}")
    #     # Step 2: Compute risk score differences (risk_diff)
    #     risk_diff = risk_scores.unsqueeze(1) - risk_scores.unsqueeze(0)
    #     print(f"Risk Score Differences: {risk_diff}")
    #     # Step 3: Time comparison matrix (t_i < t_j)
    #     t = t.unsqueeze(1)  # 确保 t 是二维张量 (batch_size, 1)
    #     time_comparison = torch.triu(torch.sign((one_vector @ t.T) - (t @ one_vector.T)), diagonal=1)
    #     print(f"Time Comparison Matrix: {time_comparison}")
    #     # Step 4: Exponential ranking penalty
    #     eta_event = torch.mean(time_comparison * torch.exp(-risk_diff), dim=1, keepdim=True)
    #     print(f"Eta Event: {eta_event}")
    #     return eta_event.mean()
    def rank_loss(self,outputs, t, fc_mask2):
        """
        Pairwise Rank Loss implementation.
        - outputs: model's logits (before converting to risk scores)
        - t: survival times
        - fc_mask2: censorship mask (1 for event occurred, 0 for censored)
        """
        # Step 1: Convert logits to risk scores
        risk_scores = -torch.sum(torch.cumprod(1 - torch.sigmoid(outputs), dim=1), dim=1)
        
        loss = 0.0
        count = 0
        for i in range(len(t)):
            if fc_mask2[i] == 1:  # Only consider uncensored events
                # Step 2: Find all j where t[j] > t[i]
                relevant_indices = (t > t[i]).nonzero(as_tuple=True)[0]
                if len(relevant_indices) > 0:
                    loss += torch.logsumexp(risk_scores[relevant_indices], dim=0) - risk_scores[i]
                    count += 1
        
        # Step 3: Return the mean loss
        if count > 0:
            return loss / count
        else:
            return torch.tensor(0.0, device=outputs.device)