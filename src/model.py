"""
Advanced PMSM Fault Detection Model
====================================
Multi-Branch CNN + BiLSTM + Channel Attention + Residual Connections
Outputs raw logits — use nn.CrossEntropyLoss() for training,
F.softmax(logits, dim=-1) for inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, filters: int, reduction: int = 8):
        super().__init__()
        mid = max(filters // reduction, 4)
        self.fc1 = nn.Linear(filters, mid)
        self.fc2 = nn.Linear(mid, filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = x.mean(dim=-1)                      # (B, C)
        att = F.relu(self.fc1(att))
        att = torch.sigmoid(self.fc2(att))
        return x * att.unsqueeze(-1)              # (B, C, L)


class TemporalAttention(nn.Module):
    """Additive temporal attention over BiLSTM sequence."""
    def __init__(self, units: int):
        super().__init__()
        self.W = nn.Linear(units, units, bias=False)
        self.u = nn.Linear(units, 1,     bias=False)

    def forward(self, x: torch.Tensor):
        score = torch.tanh(self.W(x))             # (B, T, units)
        alpha = F.softmax(self.u(score), dim=1)   # (B, T, 1)
        return (alpha * x).sum(dim=1), alpha       # (B, D), (B, T, 1)


class CNNBranch(nn.Module):
    """
    1D-CNN branch with residual connections + optional channel attention.
    Input : (B, window_size, n_branch_feats)  [seq-first]
    Output: (B, window_size, filters[-1])     [seq-first]
    """
    def __init__(self, n_branch: int, filters: list,
                 use_attention: bool, use_residual: bool):
        super().__init__()
        self.use_residual  = use_residual
        self.use_attention = use_attention
        self.filters       = list(filters)
        self.in_ch_list    = [n_branch] + list(filters[:-1])

        self.convs       = nn.ModuleList()
        self.bns         = nn.ModuleList()
        self.drops       = nn.ModuleList()
        self.projections = nn.ModuleDict()

        for i, f in enumerate(self.filters):
            in_ch = self.in_ch_list[i]
            self.convs.append(nn.Conv1d(in_ch, f, kernel_size=3, padding=1))
            self.bns.append(nn.BatchNorm1d(f))
            self.drops.append(nn.Dropout(0.2))
            if use_residual and in_ch != f:
                self.projections[str(i)] = nn.Conv1d(in_ch, f, kernel_size=1)

        if use_attention:
            self.channel_attn = ChannelAttention(self.filters[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)                    # (B,L,C) -> (B,C,L)
        for i, (conv, bn, drop) in enumerate(
                zip(self.convs, self.bns, self.drops)):
            shortcut = x
            in_ch    = self.in_ch_list[i]
            f        = self.filters[i]
            x        = drop(F.relu(bn(conv(x))))
            if self.use_residual:
                proj_key = str(i)
                if in_ch == f:
                    x = x + shortcut
                elif proj_key in self.projections:
                    x = x + self.projections[proj_key](shortcut)
            if self.use_attention and i == len(self.filters) - 1:
                x = self.channel_attn(x)
        return x.permute(0, 2, 1)                 # (B,C,L) -> (B,L,C)


class PMSMFaultDetector(nn.Module):
    """
    Multi-Branch CNN + BiLSTM.
    forward(inputs) where inputs is a list[Tensor], one per branch.
    Each tensor shape: (B, window_size, n_branch_feats)
    Returns: (B, num_classes) raw logits.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        mc           = cfg["model"]
        features     = cfg["data"]["features"]
        num_classes  = mc["num_classes"]
        filters      = mc["filters"]
        lstm_units   = mc["lstm_units"]
        self.use_att = mc.get("attention", True)
        dropout      = mc.get("dropout", 0.4)
        use_res      = mc.get("residual",  True)
        branches_cfg = mc.get("branches", {})
        feat_idx     = {f: i for i, f in enumerate(features)}

        self.branch_indices = []
        self.cnn_branches   = nn.ModuleList()
        total_cnn_ch        = 0

        for _, branch_feats in branches_cfg.items():
            idxs = [feat_idx[f] for f in branch_feats if f in feat_idx]
            if not idxs:
                continue
            self.cnn_branches.append(
                CNNBranch(len(idxs), filters, self.use_att, use_res))
            self.branch_indices.append(idxs)
            total_cnn_ch += filters[-1]

        self.bilstm = nn.LSTM(
            input_size=total_cnn_ch,
            hidden_size=lstm_units,
            batch_first=True,
            bidirectional=True,
        )
        bilstm_out      = lstm_units * 2
        self.bilstm_bn  = nn.BatchNorm1d(bilstm_out)

        if self.use_att:
            self.temporal_attn = TemporalAttention(bilstm_out)

        self.head = nn.Sequential(
            nn.Linear(bilstm_out, 256), nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: list) -> torch.Tensor:
        outs   = [b(x) for b, x in zip(self.cnn_branches, inputs)]
        merged = torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]
        x, _   = self.bilstm(merged)
        x      = self.bilstm_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.use_att:
            x, _ = self.temporal_attn(x)
        else:
            x = x.mean(dim=1)
        return self.head(x)


def build_cnn_bilstm(cfg: dict):
    """Return (model, branch_indices)."""
    features   = cfg["data"]["features"]
    feat_idx   = {f: i for i, f in enumerate(features)}
    branches   = cfg["model"].get("branches", {})
    branch_idx = [
        [feat_idx[f] for f in feats if f in feat_idx]
        for feats in branches.values()
        if any(f in feat_idx for f in feats)
    ]
    return PMSMFaultDetector(cfg), branch_idx


# ── Baseline models ───────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        nf = len(cfg["data"]["features"])
        nc = cfg["model"]["num_classes"]
        self.convs = nn.Sequential(
            nn.Conv1d(nf,  64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm1d(64),  nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm1d(128), nn.MaxPool1d(2),
            nn.Conv1d(128, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm1d(64),  nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, nc),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.convs(x.permute(0,2,1)).mean(dim=-1))


class SimpleLSTM(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        nf = len(cfg["data"]["features"])
        nc = cfg["model"]["num_classes"]
        self.l1   = nn.LSTM(nf,  128, batch_first=True)
        self.l2   = nn.LSTM(128,  64, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, nc),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _        = self.l1(x)
        _, (h_n, _) = self.l2(x)
        return self.head(h_n.squeeze(0))


def build_simple_cnn(cfg):  return SimpleCNN(cfg),  None
def build_simple_lstm(cfg): return SimpleLSTM(cfg), None


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.io_utils import load_config
    cfg = load_config()
    model, bidx = build_cnn_bilstm(cfg)
    ws = cfg["windowing"]["window_size"]
    dummy = [torch.randn(2, ws, len(i)) for i in bidx]
    out = model(dummy)
    print("Output shape :", out.shape)
    print("Softmax sum  :", out.softmax(-1).sum(-1))