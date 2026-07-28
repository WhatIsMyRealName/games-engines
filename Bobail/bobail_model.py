import json
import math
import os
import tempfile
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bobail_config import TORCH_INFERENCE_THREADS, TORCH_INTEROP_THREADS

_TORCH_INFERENCE_THREADS_CONFIGURED = False

def configure_torch_inference_threads() -> None:
    global _TORCH_INFERENCE_THREADS_CONFIGURED
    if _TORCH_INFERENCE_THREADS_CONFIGURED:
        return
    torch.set_num_threads(TORCH_INFERENCE_THREADS)
    torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
    _TORCH_INFERENCE_THREADS_CONFIGURED = True

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.activation(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.activation(y + residual)

class PolicyHead(nn.Module):
    def __init__(self, channels: int, board_size: int) -> None:
        super().__init__()
        self.board_size = board_size
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 8 * 8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        logits = self.head(x)
        logits = logits.view(batch_size, 8, 8, self.board_size, self.board_size)
        logits = logits.permute(0, 1, 3, 4, 2).contiguous()
        return logits.view(batch_size, 8 * self.board_size * self.board_size * 8)

class ValueHead(nn.Module):
    def __init__(self, channels: int, board_size: int) -> None:
        super().__init__()
        hidden_size = 128
        self.head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.Linear(32 * board_size * board_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

class _Network(nn.Module):
    def __init__(
        self,
        board_size: int = 5,
        num_channels: int = 64,
        num_res_blocks: int = 8,
        value_loss_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.value_loss_weight = value_loss_weight
        self.logs: list[float] = []
        self.logs_test: list[float] = []
        self.training_metrics: list[dict[str, float]] = []
        self.device_str = "cpu"

        self.stem = ConvBlock(3, num_channels)
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )
        self.pi_head = PolicyHead(num_channels, board_size)
        self.value_head = ValueHead(num_channels, board_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(0)

        features = self.stem(x)
        features = self.residual_tower(features)
        return self.pi_head(features), self.value_head(features)

    @staticmethod
    def load(base_path: str) -> "_Network":
        if not os.path.exists(base_path):
            raise FileNotFoundError(base_path)
        model = _Network()
        model.load_state_dict(torch.load(base_path, map_location="cpu"))

        meta_path = base_path + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta: dict[str, Any] = json.load(fh)
            for key in ["tau", "learning_rate", "logs", "logs_test", "training_metrics", "value_loss_weight"]:
                if key in meta:
                    setattr(model, key, meta[key])
        return model

    def save(self, base_path: str) -> None:
        os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)
        torch.save(self.state_dict(), base_path)

    def export(self) -> None:
        self.eval()

    def summary(self, theory: float | None = None) -> None:
        import matplotlib.pyplot as plt

        plt.plot(self.logs, marker=".", linestyle="-", color="b", label="Erreur train")
        plt.plot(self.logs_test, marker=".", linestyle="-", color="r", label="Erreur validation")
        if theory is not None:
            plt.hlines(theory, 0, len(self.logs) - 1, colors="g", linestyles="dashed", label="Erreur theorique aleatoire")
        plt.title("Erreur au cours du temps (epochs)", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Erreur (UA)", fontsize=12)
        plt.grid(True)
        plt.ylim(bottom=0)
        plt.legend(loc="best")
        plt.show()

    @staticmethod
    def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        return value.detach().cpu().numpy()

    @staticmethod
    def _make_loader(
        x: np.ndarray | torch.Tensor,
        y_pi: np.ndarray | torch.Tensor,
        y_v: np.ndarray | torch.Tensor,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        x_t = torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x.float()
        y_pi_t = torch.from_numpy(y_pi).float() if isinstance(y_pi, np.ndarray) else y_pi.float()
        y_v_t = torch.from_numpy(y_v).float() if isinstance(y_v, np.ndarray) else y_v.float()
        return DataLoader(TensorDataset(x_t, y_pi_t, y_v_t), batch_size=batch_size, shuffle=shuffle)

def debug_network_shapes() -> None:
    test_data = torch.zeros((1, 3, 5, 5), dtype=torch.float32)
    test_data[0, 0, 4, :] = 1.0
    test_data[0, 1, 2, 2] = 1.0
    test_data[0, 2, 0, :] = 1.0

    net = _Network()
    net.eval()
    with torch.no_grad():
        pi, v = net(test_data)

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Policy shape: {tuple(pi.shape)}")
    print(f"Value shape: {tuple(v.shape)}")
    print(f"Initial value: {v.item():.4f}")


if __name__ == "__main__":
    debug_network_shapes()
