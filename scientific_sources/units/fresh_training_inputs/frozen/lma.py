import argparse
import itertools
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# ==========================================
# 0. Device & Reproducibility
# ==========================================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)
print(f"Running on: {DEVICE}")


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# ==========================================
# 1. Dataset: N-bit Parity
# ==========================================
class ParityDataset(Dataset):
    """
    Synthetic dataset for the N-bit Parity problem.
    Input: Binary sequence of length N (e.g., [1, 0, 1, 1]).
    Output: 1 if the sum of bits is odd, 0 if even.
    """

    def __init__(self, num_samples: int, seq_len: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        # Generate random binary data
        self.data = torch.randint(0, 2, (num_samples, seq_len)).float()
        # Compute parity: sum modulo 2
        self.labels = (self.data.sum(dim=1) % 2).long()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ==========================================
# 2. Baseline: Standard Transformer Encoder
# ==========================================
class StandardTransformer(nn.Module):
    """
    Standard Transformer Encoder baseline.
    Represents the pairwise attention capability under fixed width/depth.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        n_heads: int,
        depth: int,
        dropout: float = 0.1,
        use_positional: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.use_positional = use_positional
        if use_positional:
            self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.pool = nn.AdaptiveAvgPool1d(1)  # Pooling over sequence (invariant head)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N) -> (B, N, 1)
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        if self.use_positional:
            x = x + self.pos_embedding                              # (B, N, D)
        x = self.transformer_encoder(x)                             # (B, N, D)

        # Global average pooling over positions
        x = x.transpose(1, 2)                                       # (B, D, N)
        x = self.pool(x).squeeze(-1)                               # (B, D)
        return self.classifier(x)                                  # (B, 2)


# ==========================================
# 2b. Deep Sets Baseline
# ==========================================
class DeepSets(nn.Module):
    """
    Sum-decomposable Deep Sets: rho(sum phi(x_i)).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)              # (B, N, 1)
        phi_x = self.phi(x)              # (B, N, D)
        pooled = phi_x.sum(dim=1)        # (B, D)
        return self.rho(pooled)          # (B, 2)


# ==========================================
# 3. Latent Möbius Attention (LMA)
# ==========================================
class LatentMobiusAttention(nn.Module):
    """
    Latent Möbius Attention (single head).

    - Tokens are softly routed into M latent buckets.
    - For each order r=1..k, we form subset-based interactions between buckets:
        gamma^{(r)}_{j1,...,jr} = ⊙_{ℓ=1..r} W^{(r,ℓ)} Z_{j_ℓ}
      where Z_j is the latent representation of bucket j.
    - These interaction features are then attended to by the original token queries.
    """

    def __init__(self, d_model: int, num_buckets_M: int, order_k: int, d_latent: int):
        super().__init__()
        self.d_model = d_model
        self.M = num_buckets_M
        # Ensure k ≤ M so that combinations are non-empty
        self.k = min(order_k, num_buckets_M)
        self.d_latent = d_latent

        # --- Token projections ---
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_latent)

        # Routing into buckets (soft hashing)
        self.W_H = nn.Linear(d_model, num_buckets_M)

        # --- Interaction projections ---
        # For each order r, we project Z into r "legs" of latent dimension d_latent
        self.interaction_projs = nn.ModuleList()
        for r in range(1, self.k + 1):
            # Single Linear produces r * d_latent, later chunked into r parts
            self.interaction_projs.append(nn.Linear(d_latent, r * d_latent))

        # --- Interaction MLPs ---
        # Each order-r interaction feature (latent d_latent) is mapped to d_model
        self.interaction_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_latent),
                    nn.Linear(d_latent, d_latent),
                    nn.GELU(),
                    nn.Linear(d_latent, d_model),
                )
                for _ in range(self.k)
            ]
        )

        # Learnable gates controlling the contribution of each order
        self.order_gates = nn.Parameter(torch.ones(self.k))

        # --- Readout ---
        self.W_out = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        # Precompute (unordered, unique) combinations of bucket indices for each order
        # r = 1: (0), (1), ..., (M-1)
        # r = 2: (0,1), (0,2), ...
        self.combos = {}
        for r in range(1, self.k + 1):
            idx = list(itertools.combinations(range(self.M), r))
            self.combos[r] = torch.tensor(idx, dtype=torch.long)  # (#comb_r, r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D_model)
        returns: (B, N, D_model) after Möbius-style attention + residual + LayerNorm
        """
        B, N, D = x.shape
        device = x.device

        # 1. Token projections
        Q = self.W_q(x)                  # (B, N, D)
        K = self.W_k(x)                  # (B, N, D)
        V = self.W_v(x)                  # (B, N, d_latent)

        # 2. Differentiable latent bucketing: pi_i(j) = softmax(W_H K_i)[j]
        u = self.W_H(K)                  # (B, N, M)
        pi = F.softmax(u, dim=-1)        # (B, N, M)

        # Z_j = Σ_i pi_i(j) v_i  (latent bucket values)
        Z = torch.bmm(pi.transpose(1, 2), V)  # (B, M, d_latent)

        # 3. Latent Möbius interactions for orders r = 1..k
        interaction_outputs = []

        for r in range(1, self.k + 1):
            # Project Z into r latent "legs": (B, M, r*d_latent) -> r×(B, M, d_latent)
            proj = self.interaction_projs[r - 1](Z)      # (B, M, r*d_latent)
            parts = proj.chunk(r, dim=-1)                # list of r tensors

            combos_r = self.combos[r].to(device)         # (#comb_r, r)

            # For each leg i, gather bucket projections at indices combos_r[:, i]
            gathered = []
            for i in range(r):
                idx_i = combos_r[:, i]                   # (#comb_r,)
                # parts[i]: (B, M, d_latent) → (B, #comb_r, d_latent)
                gathered_i = parts[i][:, idx_i, :]
                gathered.append(gathered_i)

            # Elementwise product across r legs (Hadamard product)
            gamma_r = gathered[0]                        # (B, #comb_r, d_latent)
            for i in range(1, r):
                gamma_r = gamma_r * gathered[i]          # (B, #comb_r, d_latent)

            # Order-r interaction features in model space
            feat_r = self.interaction_mlps[r - 1](gamma_r)   # (B, #comb_r, d_model)
            feat_r = self.order_gates[r - 1] * feat_r        # scale by learnable gate
            interaction_outputs.append(feat_r)

        # Concatenate all orders into a single "interaction memory"
        # Memory: (B, L_total, d_model), where L_total = Σ_r C(M, r)
        Memory = torch.cat(interaction_outputs, dim=1)

        # 4. Query-based readout: each token attends over the interaction memory
        scores = torch.bmm(Q, Memory.transpose(1, 2)) / math.sqrt(self.d_model)  # (B, N, L_total)
        attn_weights = F.softmax(scores, dim=-1)
        readout = torch.bmm(attn_weights, Memory)                                # (B, N, d_model)

        # 5. Residual connection + LayerNorm
        out = self.W_out(readout)                                               # (B, N, d_model)
        return self.layer_norm(x + out)


class LMANetwork(nn.Module):
    """
    Full LMA-based network:
      - input embedding + positional encoding
      - stack of LMA blocks with FFN residual
      - global pooling + classifier
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        num_buckets: int,
        order_k: int,
        d_latent: int,
        depth: int,
        use_positional: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.use_positional = use_positional
        if use_positional:
            self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        self.layers = nn.ModuleList(
            [
                LatentMobiusAttention(d_model, num_buckets, order_k, d_latent)
                for _ in range(depth)
            ]
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N) -> (B, N, 1)
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        if self.use_positional:
            x = x + self.pos_embedding                      # (B, N, D)

        # Stack LMA blocks with FFN residuals
        for layer in self.layers:
            x = layer(x)                                    # (B, N, D)
            x = x + self.ffn(x)                             # (B, N, D)

        # Set-level classification head (global pooling)
        x = x.transpose(1, 2)                               # (B, D, N)
        x = self.pool(x).squeeze(-1)                        # (B, D)
        return self.classifier(x)                           # (B, 2)


# ==========================================
# 4. Training Engine
# ==========================================
def train_model(model, train_loader, val_loader, epochs: int, name: str):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    print(f"\nTraining {name}...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            # Clip gradients to stabilize high-order terms
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        train_acc = train_correct / train_total
        acc = correct / total
        history["train_loss"].append(total_loss / len(train_loader))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(acc)

        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Ep {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {acc:.4f}"
            )

    elapsed = time.time() - start_time
    print(f"{name} finished in {elapsed:.2f}s")
    return history


def evaluate_model(model, data_loader):
    """
    Evaluate a trained classifier and return (accuracy, AUC).
    """
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(dim=1)
            probs = torch.softmax(logits, dim=1)[:, 1]

            correct += (preds == y).sum().item()
            total += y.size(0)
            all_labels.append(y.cpu())
            all_probs.append(probs.cpu())

    acc = correct / total if total else 0.0
    if all_labels:
        labels_np = torch.cat(all_labels).numpy()
        probs_np = torch.cat(all_probs).numpy()
        if len(np.unique(labels_np)) > 1:
            auc = roc_auc_score(labels_np, probs_np)
        else:
            auc = float("nan")
    else:
        auc = float("nan")

    return acc, auc


# ==========================================
# 5. Main Experiment: Parity
# ==========================================
def run_experiments(args):
    # Configuration aligned with set-mode parity in the paper
    N_LIST = [10]           # default parity size
    NUM_SAMPLES = 6000
    BATCH_SIZE = 64
    EPOCHS = 100

    D_MODEL = 32
    D_LATENT = 16
    MAX_BUCKETS = 16
    LMA_DEPTH = 1
    HEADS = 4

    SEED = args.seed

    if args.models.lower() == "all":
        selected_models = {"deepsets", "tx1", "tx2", "tx3", "lma3", "lma4", "lma5"}
    else:
        selected_models = set(m.strip().lower() for m in args.models.split(","))

    # fix dataset generation and split seeding
    set_seed(SEED)

    for SEQ_LEN in N_LIST:
        M_BUCKETS = min(MAX_BUCKETS, SEQ_LEN)

        # Prepare Data
        dataset = ParityDataset(NUM_SAMPLES, SEQ_LEN)
        test_size = int(0.1 * NUM_SAMPLES)
        val_size = int(0.1 * NUM_SAMPLES)
        train_size = NUM_SAMPLES - val_size - test_size
        if train_size <= 0:
            raise ValueError("Not enough samples for train split; adjust NUM_SAMPLES or split sizes.")
        generator = torch.Generator().manual_seed(SEED)
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )

        def make_loaders(seed: int):
            g = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
            return train_loader, val_loader, test_loader

        results = {}
        test_results = []

        def log_test_metrics(name: str, model: nn.Module, test_loader):
            acc, auc = evaluate_model(model, test_loader)
            test_results.append((name, acc, auc))
            print(f"[{name}] Test Acc: {acc:.4f} | Test AUC: {auc:.4f}")

        # --- Deep Sets ---
        if "deepsets" in selected_models:
            set_seed(SEED)
            train_loader, val_loader, test_loader = make_loaders(SEED)
            model_ds = DeepSets(d_model=D_MODEL)
            results["DeepSets"] = train_model(
                model_ds, train_loader, val_loader, EPOCHS, "DeepSets"
            )
            log_test_metrics("DeepSets", model_ds, test_loader)

        # --- Transformer baselines (no positional encodings), depths 3-5 ---
        transformer_depths = [(1, "tx1"), (2, "tx2"), (3, "tx3")]
        for depth, key in transformer_depths:
            if key not in selected_models:
                continue
            set_seed(SEED)
            train_loader, val_loader, test_loader = make_loaders(SEED)
            model_tx = StandardTransformer(
                SEQ_LEN, D_MODEL, n_heads=HEADS, depth=depth, use_positional=False
            )
            name = f"Transformer (D={depth}, no pos)"
            results[name] = train_model(model_tx, train_loader, val_loader, EPOCHS, name)
            log_test_metrics(name, model_tx, test_loader)

        # --- LMA variants with higher k ---
        lma_orders = [(3, "lma3"), (4, "lma4"), (5, "lma5")]
        for order_k, key in lma_orders:
            if key not in selected_models:
                continue
            set_seed(SEED)
            train_loader, val_loader, test_loader = make_loaders(SEED)
            model_lma = LMANetwork(
                seq_len=SEQ_LEN,
                d_model=D_MODEL,
                num_buckets=M_BUCKETS,
                order_k=order_k,
                d_latent=D_LATENT,
                depth=LMA_DEPTH,
                use_positional=False,
            )
            name = f"LMA (k={order_k}, depth={LMA_DEPTH})"
            results[name] = train_model(
                model_lma, train_loader, val_loader, EPOCHS, name
            )
            log_test_metrics(name, model_lma, test_loader)

        # --- Plotting ---
        if results:
            plt.figure(figsize=(10, 5))
            for name, hist in results.items():
                plt.plot(hist["val_acc"], label=name, linewidth=2)
            plt.title(f"{SEQ_LEN}-bit Parity Validation Accuracy (Set-mode)")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend(loc="lower right", fontsize=8)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            output_path = f"Figure_parity_N{SEQ_LEN}.png"
            plt.savefig(output_path, dpi=200)
            plt.close()
            print(f"Saved validation curves to {output_path}")

        if test_results:
            print("\nHeld-out test metrics:")
            for name, acc, auc in test_results:
                print(f"- {name}: Acc={acc:.4f}, AUC={auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", "--model",
        type=str,
        default="all",
        help="Comma-separated subset of models to run "
             "(deepsets,tx3,tx4,tx5,lma3,lma4,lma5) or 'all'.",
    )
    parser.add_argument("--seed", type=int, default=41, help="Global seed for repeatability.")
    args = parser.parse_args()
    run_experiments(args)
