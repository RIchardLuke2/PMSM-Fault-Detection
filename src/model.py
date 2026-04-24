"""
Advanced PMSM Fault Detection Model
====================================
Architecture: Multi-Branch CNN + BiLSTM with Channel Attention + Residual Connections

Branch layout (per the config):
  • current_branch   : Ia, Ib           → CNN feature extractor
  • voltage_branch   : VDC, VD          → CNN feature extractor
  • thermal_branch   : T1, T2, T3       → CNN feature extractor
  • dc_branch        : IDC              → CNN feature extractor

All branches concat → BiLSTM → Attention → Dense → Softmax (9 classes)
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# ── Attention layer ───────────────────────────────────────────────────────────

class ChannelAttention(layers.Layer):
    """Squeeze-and-Excitation style channel attention."""
    def __init__(self, filters: int, reduction: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.gap = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(max(filters // reduction, 4), activation="relu")
        self.fc2 = layers.Dense(filters, activation="sigmoid")
        self.reshape = layers.Reshape((1, filters))

    def call(self, x):
        att = self.gap(x)          # (B, F)
        att = self.fc1(att)
        att = self.fc2(att)        # (B, F) gate
        att = self.reshape(att)    # (B, 1, F)
        return x * att             # broadcast


class TemporalAttention(layers.Layer):
    """Additive temporal attention over BiLSTM output sequence."""
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.W = layers.Dense(units, use_bias=False)
        self.u = layers.Dense(1, use_bias=False)

    def call(self, x):
        # x: (B, T, D)
        score = tf.nn.tanh(self.W(x))   # (B, T, units)
        score = self.u(score)            # (B, T, 1)
        alpha = tf.nn.softmax(score, axis=1)  # (B, T, 1)
        context = tf.reduce_sum(alpha * x, axis=1)  # (B, D)
        return context, alpha


# ── Branch builder ────────────────────────────────────────────────────────────

def _build_cnn_branch(inp: tf.Tensor, filters: list, l2: float,
                      name: str, use_attention: bool, use_residual: bool) -> tf.Tensor:
    """
    1D-CNN branch: stacked Conv→BN→ReLU→Dropout blocks with optional
    residual connection and channel attention.
    """
    x = inp
    for i, f in enumerate(filters):
        shortcut = x
        x = layers.Conv1D(f, kernel_size=3, padding="same",
                          kernel_regularizer=regularizers.l2(l2),
                          name=f"{name}_conv{i+1}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i+1}")(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.2)(x)

        if use_residual and shortcut.shape[-1] == f:
            x = layers.Add(name=f"{name}_res{i+1}")([x, shortcut])
        elif use_residual:
            # Projection shortcut
            shortcut = layers.Conv1D(f, 1, padding="same",
                                     name=f"{name}_proj{i+1}")(shortcut)
            x = layers.Add(name=f"{name}_res_proj{i+1}")([x, shortcut])

        if use_attention and i == len(filters) - 1:
            x = ChannelAttention(f, name=f"{name}_ca")(x)

    return x


# ── Main model builder ────────────────────────────────────────────────────────

def build_cnn_bilstm(cfg: dict) -> keras.Model:
    mc = cfg["model"]
    wc = cfg["windowing"]
    features: list = cfg["data"]["features"]
    window_size: int = wc["window_size"]
    n_features: int = len(features)
    num_classes: int = mc["num_classes"]
    filters: list = mc["filters"]
    lstm_units: int = mc["lstm_units"]
    use_attention: bool = mc.get("attention", True)
    use_residual: bool = mc.get("residual", True)
    dropout: float = mc.get("dropout", 0.4)
    l2: float = mc.get("l2_reg", 1e-4)

    # Branch feature indices
    branches = mc.get("branches", {})
    feat_idx = {f: i for i, f in enumerate(features)}

    branch_inputs, branch_outputs = [], []

    for branch_name, branch_feats in branches.items():
        idxs = [feat_idx[f] for f in branch_feats if f in feat_idx]
        if not idxs:
            continue
        n_branch = len(idxs)
        inp = keras.Input(shape=(window_size, n_branch), name=f"input_{branch_name}")
        branch_inputs.append((inp, idxs))
        out = _build_cnn_branch(inp, filters, l2, branch_name,
                                use_attention, use_residual)
        branch_outputs.append(out)

    # Merge branches
    if len(branch_outputs) > 1:
        merged = layers.Concatenate(name="branch_merge")(branch_outputs)
    else:
        merged = branch_outputs[0]

    # BiLSTM
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True,
                    kernel_regularizer=regularizers.l2(l2)),
        name="bilstm")(merged)
    x = layers.BatchNormalization()(x)

    # Temporal attention
    if use_attention:
        att_layer = TemporalAttention(lstm_units * 2, name="temporal_attention")
        x, _ = att_layer(x)
    else:
        x = layers.GlobalAveragePooling1D()(x)

    # Classification head
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout / 2)(x)
    output = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=[inp for inp, _ in branch_inputs],
                        outputs=output, name="PMSM_FDD_CNN_BiLSTM")
    return model, [idxs for _, idxs in branch_inputs]


def build_simple_cnn(cfg: dict) -> keras.Model:
    """Simple baseline CNN (single branch, no attention)."""
    mc = cfg["model"]
    wc = cfg["windowing"]
    n_features = len(cfg["data"]["features"])
    inp = keras.Input(shape=(wc["window_size"], n_features), name="input")
    x = inp
    for f in [64, 128, 64]:
        x = layers.Conv1D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(mc["num_classes"], activation="softmax")(x)
    return keras.Model(inp, out, name="SimpleCNN"), None


def build_simple_lstm(cfg: dict) -> keras.Model:
    """Simple baseline LSTM (single branch)."""
    mc = cfg["model"]
    wc = cfg["windowing"]
    n_features = len(cfg["data"]["features"])
    inp = keras.Input(shape=(wc["window_size"], n_features), name="input")
    x = layers.LSTM(128, return_sequences=True)(inp)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(mc["num_classes"], activation="softmax")(x)
    return keras.Model(inp, out, name="SimpleLSTM"), None


if __name__ == "__main__":
    from utils.io_utils import load_config
    cfg = load_config()
    model, _ = build_cnn_bilstm(cfg)
    model.summary()
