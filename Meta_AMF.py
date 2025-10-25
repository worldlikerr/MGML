import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MetaNetwork(nn.Module):
    def __init__(self, input_features_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_features_dim, 64)
        self.relu = torch.nn.ReLU()
        self.output_layer = nn.Linear(64, 5)  # T1, T2, fusion_balance_weight, beta_param, alpha_param

        nn.init.constant_(self.output_layer.bias[0], 1.2528)
        nn.init.constant_(self.output_layer.bias[1], 2.0794)
        nn.init.constant_(self.output_layer.bias[2], 0.0)
        nn.init.constant_(self.output_layer.bias[3], 3.89)
        nn.init.constant_(self.output_layer.bias[4], 3.89)

    def forward(self, features):
        x = self.fc1(features)
        x = self.relu(x)
        params = self.output_layer(x)
        T1 = torch.sigmoid(params[:, 0:1]) * (0.95 - 0.05) + 0.05
        T2 = torch.sigmoid(params[:, 1:2]) * (0.95 - 0.05) + 0.05
        fusion_balance_weight = torch.sigmoid(params[:, 2:3])

        min_beta_alpha_val = 1.0
        max_beta_alpha_val = 100.0

        beta_param = torch.sigmoid(params[:, 3:4]) * (max_beta_alpha_val - min_beta_alpha_val) + min_beta_alpha_val
        alpha_param = torch.sigmoid(params[:, 4:5]) * (max_beta_alpha_val - min_beta_alpha_val) + min_beta_alpha_val

        return T1, T2, fusion_balance_weight, beta_param, alpha_param

def smooth_max(tensors, beta=50.0):
    max_val = tensors.max(dim=1, keepdim=True)[0]
    shifted_tensors = tensors - max_val
    log_sum_exp_term = torch.logsumexp(beta * shifted_tensors, dim=1, keepdim=True)
    smooth_max_val = (1.0 / beta) * log_sum_exp_term + max_val
    return smooth_max_val.squeeze(1)

def smooth_min(tensors, alpha=50.0):
    max_val = tensors.max(dim=1, keepdim=True)[0]
    shifted_tensors = tensors - max_val
    log_sum_exp_term = torch.logsumexp(-alpha * shifted_tensors, dim=1, keepdim=True)
    smooth_min_val = -(1.0 / alpha) * log_sum_exp_term + max_val
    return smooth_min_val.squeeze(1)

def Adaptive_Modality_Fuser(pre1_logits, pre2_logits, pre3_logits, pre4_logits,
                   T1_dynamic, T2_dynamic, fusion_balance_weight, beta_param, alpha_param):
    num_cls = pre1_logits.shape[1]

    fusion_balance_weight_expanded = fusion_balance_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, num_cls, -1, -1, -1)

    all_raw_logits = torch.stack([pre1_logits, pre2_logits, pre3_logits, pre4_logits], dim=1)

    high_confidence_fused_logits = smooth_max(all_raw_logits, beta=beta_param)
    conservative_fused_logits = smooth_min(all_raw_logits, alpha=alpha_param)
    final_soft_target_logits = (fusion_balance_weight_expanded * high_confidence_fused_logits) + \
                               ((1 - fusion_balance_weight_expanded) * conservative_fused_logits)

    return final_soft_target_logits