import torch

def criterion_nig(nig_params, y, lamb=0.01):
    """
    计算 Normal Inverse Gamma (NIG) 回归损失。
    Args:
        nig_params (torch.Tensor): 模型的 NIG 输出 (mu, v, alpha, beta)。
                                   形状应为 (batch_size, seq_length, 4) 或 (batch_size, 4)。
                                   如果有多余维度，需要调整。
        y (torch.Tensor): 真实目标值。形状应与 mu 匹配。
        lamb (float): 正则化项的权重。
    Returns:
        torch.Tensor: 计算得到的损失值 (标量)。
    """
    # 确保 y 的形状与 mu, v, alpha, beta 匹配
    # 假设 nig_params 的最后一维是 [mu, v, alpha, beta]
    if nig_params.dim() > y.dim():
         # 如果 nig_params 是 (batch, seq, 4) 而 y 是 (batch, seq)
         # 需要 unsqueeze y 或者调整 nig_params
         if nig_params.shape[:-1] == y.shape:
             y = y.unsqueeze(-1) # 变为 (batch, seq, 1)
         else:
             # 可能需要更复杂的 reshape 或 squeeze 操作
             # 例如，如果只关心最后一个时间步的预测：
             # nig_params = nig_params[:, -1, :] # (batch, 4)
             # y = y[:, -1].unsqueeze(-1) # (batch, 1)
             # 或者如果 y 只有一个值对应整个序列
             # y = y.view(-1, 1, 1).expand_as(nig_params[..., :1])
             raise ValueError(f"Shape mismatch: nig_params {nig_params.shape}, y {y.shape}")

    # 分割 NIG 参数
    mu, v, alpha, beta = torch.split(nig_params, 1, dim=-1)

    # 计算 NIG 负对数似然损失 (NLL)
    # 根据论文或标准 NIG 定义调整公式
    two_beta_v_plus_1 = 2 * beta * (1 + v) # om in paper? v is nu/lambda?

    # 检查 alpha > 1 (模型设计保证了 alpha > 1)
    # 检查 two_beta_v_plus_1 > 0 (模型设计保证了 beta > 0, v > 0)

    # 使用 torch.lgamma 计算 log gamma 函数
    log_gamma_alpha = torch.lgamma(alpha)
    log_gamma_alpha_plus_half = torch.lgamma(alpha + 0.5)

    # 防止 log(0) 或 log(<0)
    # softplus 保证 v, beta > 0, alpha > 1
    # two_beta_v_plus_1 理论上 > 0
    # v * (mu - y)**2 + two_beta_v_plus_1 理论上 > 0

    nll = (
        0.5 * torch.log(torch.pi / v)
        - alpha * torch.log(two_beta_v_plus_1)
        + (alpha + 0.5) * torch.log(v * (mu - y)**2 + two_beta_v_plus_1)
        + log_gamma_alpha
        - log_gamma_alpha_plus_half
    )

    # 计算正则化项 (根据论文)
    # Error = |mu - y|
    # Uncertainty = 2*v + alpha ? (需要确认论文中的精确定义)
    # 论文中似乎是 lambda * E * (2*nu + alpha)
    # 这里 v 对应 nu, lambda 对应 evidence lambda (未直接输出，可能需要从 v 推导?)
    # 假设正则化项是 lambda * |mu - y| * (2*v + alpha)
    reg = lamb * torch.abs(mu - y) * (2 * v + alpha)

    # 计算总损失 (平均 NLL + 平均正则化)
    # 需要确保在 batch 和 seq 维度上正确平均
    loss = torch.mean(nll + reg)

    # 原始代码中的实现方式 (逐元素求和再除以长度)
    # om = 2 * beta * (1 + v) # 这里用 v 替换了 la
    # loss_orig = sum(
    #     0.5 * torch.log(torch.pi / v)
    #     - alpha * torch.log(om)
    #     + (alpha + 0.5) * torch.log(v * (mu - y) ** 2 + om)
    #     + torch.lgamma(alpha)
    #     - torch.lgamma(alpha + 0.5)
    # ) / len(mu.view(-1)) # 展平后计算长度
    # lossr_orig = lamb * sum(torch.abs(mu - y) * (2 * v + alpha)) / len(mu.view(-1))
    # loss_orig = loss_orig + lossr_orig
    # return loss_orig.mean() # 确保返回标量

    return loss

