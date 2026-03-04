import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt

# -----------------------------
# Utilidades (igual idea que tu notebook)
# -----------------------------
def rank_of_positive(scores, pos_in_test=0):
    scores = np.asarray(scores)
    pos_score = scores[pos_in_test]
    # rank 1 = mejor (más score)
    return 1 + int(np.sum(scores > pos_score))

def ranking_metrics(ranks):
    ranks = np.array(ranks, dtype=float)
    return {
        "MeanRank": float(ranks.mean()),
        "MedianRank": float(np.median(ranks)),
        "MRR": float(np.mean(1.0 / ranks)),
        "Hit@10": float(np.mean(ranks <= 10)),
        "Hit@50": float(np.mean(ranks <= 50)),
        "Hit@100": float(np.mean(ranks <= 100)),
    }

def prepare_split(X, Y, W, idx_ones, idx_zeros, j=0, n_test=4000, seed=42):
    """
    Split para un virus W:
    - deja fuera 1 positivo (idx_ones[j]) y lo mete en test
    - samplea (n_test-1) negativos y los mete en test
    - train = todo lo demás
    Devuelve: x_train, y_train, x_test, y_test, pos_in_test
    """
    rng = np.random.default_rng(seed + j)

    pos_out = int(idx_ones[j])

    # ajusta n_test si no hay suficientes negativos
    max_test = 1 + len(idx_zeros)
    n_test_eff = min(n_test, max_test)
    n_neg_test = n_test_eff - 1

    if n_neg_test <= 0:
        raise ValueError("No hay negativos disponibles para armar el test. Revisa Y[:, W].")

    neg_test = rng.choice(idx_zeros, size=n_neg_test, replace=False)

    test_idx = np.concatenate([[pos_out], neg_test])

    all_idx = np.arange(X.shape[0])
    train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)

    x_train = X[train_idx]
    y_train = Y[train_idx, W].astype(int)

    x_test = X[test_idx]
    y_test = Y[test_idx, W].astype(int)

    pos_in_test = 0
    return x_train, y_train, x_test, y_test, pos_in_test

# -----------------------------
# Modelo (igual a tu notebook)
# -----------------------------
class TorchMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # logits
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class TorchMLPWrapper:
    def __init__(self, in_dim, lr=1e-3, epochs=25, batch_size=256,
                 pos_weight=27.0, weight_decay=0.0035, seed=42):
        self.in_dim = in_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.pos_weight = float(pos_weight)
        self.weight_decay = float(weight_decay)
        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TorchMLP(in_dim).to(self.device)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight, device=self.device))
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # mini-batches
        n = X.shape[0]
        idx = np.arange(n)

        self.model.train()
        history = []
        for ep in range(self.epochs):
            np.random.shuffle(idx)
            ep_loss = 0.0
            for start in range(0, n, self.batch_size):
                batch_idx = idx[start:start+self.batch_size]
                xb = X[batch_idx]
                yb = y[batch_idx]

                opt.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

                ep_loss += loss.item() * len(batch_idx)

            ep_loss /= n
            history.append(ep_loss)

        return history

    def decision_function(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(X).detach().cpu().numpy()
        return logits  # logits sirven para ranking

# -----------------------------
# Cargar datos (tus npy)
# -----------------------------
X = np.load("X_MACCS.npy")
Y = np.load("Y_final_schemnet.npy")

# W por defecto (mismo estilo que tu notebook)
rng = np.random.default_rng(42)
W_default = int(rng.integers(0, Y.shape[1]))

# -----------------------------
# Función de demo (1 split j=0)
# -----------------------------
def run_demo(W, n_test, epochs, pos_weight, weight_decay, lr, batch_size):
    W = int(W)
    n_test = int(n_test)

    idx_ones = np.where(Y[:, W] == 1)[0]
    idx_zeros = np.where(Y[:, W] == 0)[0]

    if len(idx_ones) == 0:
        return "Ese W no tiene positivos.", None, None

    # un solo split
    j = 0
    x_train, y_train, x_test, y_test, pos_in_test = prepare_split(
        X, Y, W, idx_ones, idx_zeros, j=j, n_test=n_test, seed=42
    )

    model = TorchMLPWrapper(
        in_dim=X.shape[1],
        lr=float(lr),
        epochs=int(epochs),
        batch_size=int(batch_size),
        pos_weight=float(pos_weight),
        weight_decay=float(weight_decay),
        seed=42
    )

    loss_hist = model.fit(x_train, y_train)

    scores = model.decision_function(x_test)  # logits
    r = rank_of_positive(scores, pos_in_test=pos_in_test)

    # tabla top-20 del test
    order = np.argsort(-scores)  # descendente
    topk = order[: min(20, len(order))]
    df = pd.DataFrame({
        "rank_en_test": np.arange(1, len(topk)+1),
        "idx_en_test": topk,
        "y_true": y_test[topk],
        "score_logit": scores[topk]
    })

    # gráfico loss
    fig = plt.figure()
    plt.plot(np.arange(1, len(loss_hist)+1), loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss (BCEWithLogits)")
    plt.title(f"Curva de entrenamiento (1 split j=0) - W={W}")

    # texto resumen
    summary = (
        f"W={W} | Positivos totales en W: {int(Y[:, W].sum())}\n"
        f"Split demo: j=0 | Train size: {len(y_train)} | Test size: {len(y_test)} (1 pos + {len(y_test)-1} neg)\n"
        f"Rank del positivo en el test: {r} (1 = mejor)\n"
        f"Nota: esto es SOLO demo con 1 split; las métricas finales salen de LOOCV completo."
    )

    return summary, df, fig

# -----------------------------
# UI Gradio
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## Demo (1 split j=0) — TorchMLP ranking para un virus W")

    with gr.Row():
        W_in = gr.Number(value=W_default, label="Virus W (columna en Y)")
        n_test_in = gr.Slider(50, 4000, value=4000, step=1, label="n_test (1 pos + n_neg)")
    with gr.Row():
        epochs_in = gr.Slider(5, 50, value=25, step=1, label="epochs")
        lr_in = gr.Number(value=1e-3, label="lr")
    with gr.Row():
        posw_in = gr.Number(value=27.0, label="pos_weight")
        wd_in = gr.Number(value=0.0035, label="weight_decay")
        bs_in = gr.Slider(32, 1024, value=256, step=32, label="batch_size")

    btn = gr.Button("Correr demo (entrena 1 vez y rankea)")
    out_text = gr.Textbox(label="Resumen", lines=6)
    out_df = gr.Dataframe(label="Top-20 del test (ordenado por score)")
    out_plot = gr.Plot(label="Training loss")

    btn.click(
        run_demo,
        inputs=[W_in, n_test_in, epochs_in, posw_in, wd_in, lr_in, bs_in],
        outputs=[out_text, out_df, out_plot]
    )

demo.launch()