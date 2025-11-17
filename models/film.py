"""
FiLM (Feature-wise Linear Modulation) layer used for speaker/emotion conditioning.

Supports two variants:
1. Simple (default): single Linear → gamma, beta
2. MLP: (Linear → ReLU → Dropout → Linear) → gamma, beta

Usage:
    FiLM(in_channels=512, cond_dim=512, use_mlp=False)

Original FiLM Paper: https://arxiv.org/abs/1709.07871
"""

import torch
import torch.nn as nn


class FiLM(nn.Module):

    def __init__(
        self,
        in_channels: int,
        cond_dim: int,
        use_mlp: bool = False,
        hidden_dim: int = 256,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.use_mlp = use_mlp

        # Choose variant (nonlinear vs simple)
        if use_mlp:
            # Nonlinear variant (used in AdaSpeech, Meta-StyleSpeech)
            self.net = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
                nn.Linear(hidden_dim, 2 * in_channels),
            )
            proj_layer = self.net[-1]
        else:
            # Simple linear projection variant (aligned with original paper)
            self.net = nn.Linear(cond_dim, 2 * in_channels)
            proj_layer = self.net

        # Identity initialization (gamma=1, beta=0)
        nn.init.zeros_(proj_layer.weight)
        nn.init.zeros_(proj_layer.bias)

        with torch.no_grad():
            proj_layer.bias[:in_channels].fill_(1.0)


    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Apply FiLM modulation based on the conditioning vector
        """

        if cond is None:
            return x

        gamma, beta = self.net(cond).chunk(2, dim=-1)
        
        # Safer FiLM scaling (keeps identity around 1.0)
        # - gamma starts near 1.0 (identity)
        # - beta starts near 0.0 (no shift)
        # - tanh bounds keep modulation stable
        # gamma = 1.0 + 0.1 * torch.tanh(gamma)
        # beta  = 0.1 * torch.tanh(beta)

        gamma = gamma.unsqueeze(-1)
        beta  = beta.unsqueeze(-1)

        # FiLM Diagnostics (prints every 2000 steps)
        if hasattr(self, "global_step") and self.global_step is not None:
            if self.global_step % 1000 == 0:
                avg_gamma = gamma.abs().mean().item()
                avg_beta = beta.abs().mean().item()
                avg_act = x.abs().mean().item()
                print(
                    f"[FiLM] step={self.global_step} | "
                    f"gamma={avg_gamma:.3f} | beta={avg_beta:.3f} | act={avg_act:.3f}"
                )

        return gamma * x + beta
