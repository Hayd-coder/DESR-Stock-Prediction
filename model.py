import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import sys

class NormalInvGamma(nn.Module):
    """
    Normal Inverse Gamma (NIG) 证据层。
    将输入特征映射到 NIG 分布的四个参数：mu, v, alpha, beta。
    """
    def __init__(self, in_features, out_units):
        """
        初始化 NIG 层。
        Args:
            in_features (int): 输入特征的数量。
            out_units (int): 输出单元的数量 (通常为 1，因为预测单个值)。
        """
        super().__init__()
        # 线性层将输入映射到 4 倍输出单元的数量，因为 NIG 有 4 个参数
        self.dense = nn.Linear(in_features, out_units * 4)
        self.out_units = out_units

    def evidence(self, x):
        """
        证据函数，确保 v, alpha, beta 为正。
        使用 softplus 函数。
        Args:
            x (torch.Tensor): 输入张量。
        Returns:
            torch.Tensor: 应用 softplus 后的张量。
        """
        return F.softplus(x)

    def forward(self, x):
        """
        前向传播。
        Args:
            x (torch.Tensor): 输入张量。
        Returns:
            torch.Tensor: 包含 NIG 参数 (mu, v, alpha, beta) 的张量。
        """
        out = self.dense(x)
        # 将输出分割成四个参数
        mu, logv, logalpha, logbeta = torch.split(out, self.out_units, dim=-1)
        # 应用证据函数得到正的 v, alpha, beta
        # alpha > 1 以确保方差期望存在
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        # 将四个参数连接起来作为输出
        NIG = torch.cat([mu, v, alpha, beta], dim=-1)
        return NIG


class EGRU(nn.Module):
    """
    Evidence GRU (EGRU) 模型。
    结合了 GRU 和 NIG 证据层，用于序列预测并量化不确定性。
    """
    def __init__(self, hidden_dim, seq_length, device="cpu", input_dim=5, output_dim=1):
        """
        初始化 EGRU 模型。
        Args:
            hidden_dim (int): GRU 隐藏层的维度。
            seq_length (int): 输入序列的长度。 # 已修正注释
            device (str): 模型运行的设备 ('cpu' 或 'cuda')。
            input_dim (int): 每个时间步的输入特征维度。
            output_dim (int): 每个时间步的输出维度 (通常为 1)。
        """
        super(EGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.output_dim = output_dim
        self.device = device
        # 嵌入层，将输入特征映射到隐藏维度
        self.embs = nn.Sequential(nn.Linear(input_dim, self.hidden_dim), nn.ReLU())
        # GRU 层
        self.view_grus = nn.GRU(
            input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=True
        )
        # 从 GRU 输出 (h) 计算 NIG 参数的证据层
        self.view_h_nigs = nn.Sequential(NormalInvGamma(self.hidden_dim, output_dim))
        # 从嵌入层输出 (x_emb) 计算 NIG 参数的证据层
        self.view_x_nigs = nn.Sequential(NormalInvGamma(self.hidden_dim, output_dim))

    def nig_fusion(self, nig1, nig2):
        """
        融合两个 NIG 分布。
        根据论文中的公式进行融合。
        Args:
            nig1 (torch.Tensor): 第一个 NIG 参数张量 (mu1, v1, alpha1, beta1)。
            nig2 (torch.Tensor): 第二个 NIG 参数张量 (mu2, v2, alpha2, beta2)。
        Returns:
            torch.Tensor: 融合后的 NIG 参数张量 (mu, v, alpha, beta)。
        """
        n = nig1.shape[-1] // 4 # 输出维度
        # 分割参数
        mu1, v1, alpha1, beta1 = torch.split(nig1, n, dim=-1)
        mu2, v2, alpha2, beta2 = torch.split(nig2, n, dim=-1)

        # 融合公式
        gamma1 = v1 # v 代表精度或 gamma
        gamma2 = v2

        # 添加 epsilon 防止除以零
        epsilon = 1e-6
        gamma_sum = gamma1 + gamma2 + epsilon

        mu = (gamma1 * mu1 + gamma2 * mu2) / gamma_sum
        v = gamma1 + gamma2 # 融合后的 v (精度)
        alpha = alpha1 + alpha2 + 0.5 # 融合后的 alpha
        # 融合后的 beta 计算较为复杂
        beta = (
            beta1
            + beta2
            + 0.5 * (gamma1 * gamma2) / gamma_sum * (mu1 - mu2) ** 2
        )

        return torch.cat([mu, v, alpha, beta], dim=-1)

    def forward(self, x):
        """
        前向传播。
        Args:
            x (torch.Tensor): 输入序列张量，形状为 (batch_size, seq_length, input_dim)。
        Returns:
            torch.Tensor: 每个时间步的 NIG 参数张量，形状为 (batch_size, seq_length, output_dim * 4)。
        """
        # 初始化隐藏状态
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        # 输入嵌入
        emb = self.embs(x)
        # 通过 GRU 处理序列
        h, _ = self.view_grus(emb, h0) # h 的形状: (batch_size, seq_length, hidden_dim)
        # 计算来自 GRU 输出的 NIG 参数
        nig_h = self.view_h_nigs(h)
        # 计算来自嵌入输入的 NIG 参数
        nig_x = self.view_x_nigs(emb) # nig_x 的形状: (batch_size, seq_length, output_dim*4)
        # 融合两个 NIG 分布
        nig = self.nig_fusion(nig_h, nig_x)
        return nig
