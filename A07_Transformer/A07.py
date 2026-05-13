import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence # http://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
import awkward
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

ROOT = "/home/andre/courses/AdvMl_V26/A07_Transformer"
DATA_PATH = os.path.join(ROOT, "DATA") 
    
def norm_help(dataset):
    data = dataset["data"]

    t =  awkward.to_numpy(awkward.flatten(data[:, 0, :], axis=None))
    x = awkward.to_numpy(awkward.flatten(data[:, 1, :], axis=None))
    y = awkward.to_numpy(awkward.flatten(data[:, 2, :], axis=None))

    meta = {
        "t_mean": np.mean(t),"t_std":  np.std(t),"x_mean": np.mean(x),"x_std":  np.std(x),"y_mean": np.mean(y),"y_std":  np.std(y),"xlabel_mean": np.mean(awkward.to_numpy(dataset["xpos"])),
        "xlabel_std":  np.std(awkward.to_numpy(dataset["xpos"])),"ylabel_mean": np.mean(awkward.to_numpy(dataset["ypos"])),"ylabel_std":  np.std(awkward.to_numpy(dataset["ypos"])),
    }
    return meta

def normalize_dataset(dataset, meta):
    dataset = awkward.copy(dataset) #copy so that nothing is changed in the original dataset, awkward array seems to behave strange
    data = dataset["data"]
    
    
    norm_t = (data[:, 0:1, :] - meta["t_mean"]) / meta["t_std"]
    norm_x = (data[:, 1:2, :] - meta["x_mean"]) / meta["x_std"]
    norm_y = (data[:, 2:3, :] - meta["y_mean"]) / meta["y_std"]
    
    dataset["data"] = awkward.concatenate([norm_t, norm_x, norm_y], axis=1)
    dataset["xpos"] = (dataset["xpos"] - meta["xlabel_mean"]) / meta["xlabel_std"]
    dataset["ypos"] = (dataset["ypos"] - meta["ylabel_mean"]) / meta["ylabel_std"]
    
    return dataset

def denormalize_labels(labels, meta):
    labels_denorm = labels.clone() if hasattr(labels, "clone") else labels.copy()
    labels_denorm[..., 0] = labels_denorm[..., 0] * meta["xlabel_std"] + meta["xlabel_mean"]
    labels_denorm[..., 1] = labels_denorm[..., 1] * meta["ylabel_std"] + meta["ylabel_mean"]
    return labels_denorm


def collate_fn_transformer(batch):
    """
    Custom function that defines how batches are formed.

    To process the batch items that each have a different number of hits, it is efficient
    to first concatenate all the data into a single tensor and save the lengths of each
    individual event to be able to split the data again later.

    # F: input_dim, number of features (time, x, y)
    # N: number of hits (different for each event)
    # B: batch size

    The resulting 2D tensor has the shape (B x N, F) where B is the batch size, N is the total number of hits of all events
    in the batch, and F is the number of features (time, x, y).


    Parameters
    ----------
    batch : list
        A list of dictionaries containing the data and labels for each graph.
        The data is available in the "data" key and the labels are in the "xpos" and "ypos" keys.
    Returns
    -------
    packed_data : Batch
        A batch of graph data objects.
    labels : torch.Tensor
        A tensor containing the labels for each graph.
    """
    data_list = []
    labels = []
    lengths=[]

    for b in batch:
        # this is a loop over each event within the batch
        # b["data"] is the first entry in the batch with dimensions (n_features, n_hits)
        # where the feautures are (time, x, y)
        tensordata = torch.from_numpy(b["data"].to_numpy()).T
        # the original data is in double precision (float64), for our case single precision is sufficient
        # we let's convert to single precision (float32) to save memory and computation time
        tensordata = tensordata.to(dtype=torch.float32)

        lengths.append(tensordata.shape[0])

        data_list.append(tensordata)

        # also the labels need to be packaged as pytorch tensors
        labels.append(torch.Tensor([b["xpos"], b["ypos"]]).unsqueeze(0))

    labels = torch.cat(labels, dim=0) # convert the list of tensors to a single tensor

    data_vec=torch.cat(data_list) # (B, N, F)  -> (BxN, F) where B is the batch size, N is the number of hits, and F is the number of features (time, x, y)

    ## return a list [datalist, lengths]
    return [data_vec, lengths], labels


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim=3,      # (t, x, y)
        d_model=128,       # hidden size
        nhead=2,          # ≥ 2 heads
        num_layers=2,     # ≥ 2 layers
        dim_feedforward=128,
        output_dim=2      # (xpos, ypos)
    ):
        super().__init__()

        # 1) input embedding (F -> D)
        self.input_proj = nn.Linear(input_dim, d_model)

        # transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="relu",
            batch_first=True,
            norm_first=True,
            dropout=0.02
        )

        # stack layers
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 2) output projection (D -> 2)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, data) -> torch.Tensor:
        src, lengths = data  # (BxN, F), list of lengths

        # 1) embed
        src = self.input_proj(src)  # (BxN, D)

        # 2) split into sequences
        parts = src.split(lengths, dim=0)

        # 3) pad
        padded = pad_sequence(parts, batch_first=True)  # (B, max_len, D)
        batch_size, max_len, _ = padded.shape

        # 4) mask
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=padded.device)
        for i, L in enumerate(lengths):
            mask[i, L:] = True

        # 5) transformer
        enc_out = self.encoder(padded, src_key_padding_mask=mask)

        # 6) masked mean pooling
        valid_mask = ~mask
        summed = (enc_out * valid_mask.unsqueeze(-1)).sum(dim=1)

        lengths_tensor = torch.tensor(lengths, device=enc_out.device).unsqueeze(1)
        pooled = summed / lengths_tensor

        # 7) output
        out = self.output_proj(pooled)  # (B, 2)

        return out


def train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-3):
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # ---- TRAIN ----
        model.train()
        running_loss = 0.0

        for data, labels in tqdm(train_loader):
            # move to device
            data_vec, lengths = data
            data_vec = data_vec.to(device)
            labels = labels.to(device)

            # forward
            preds = model([data_vec, lengths])

            loss = F.mse_loss(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # ---- VALIDATION ----
        model.eval()
        val_running = 0.0

        with torch.no_grad():
            for data, labels in val_loader:
                data_vec, lengths = data
                data_vec = data_vec.to(device)
                labels = labels.to(device)

                preds = model([data_vec, lengths])
                loss = F.mse_loss(preds, labels)

                val_running += loss.item()

        val_loss = val_running / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    return train_losses, val_losses

def evaluate_model(model, test_loader, device, meta):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data_vec, lengths = data
            data_vec = data_vec.to(device)
            labels = labels.to(device)

            preds = model([data_vec, lengths])

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # denormalize
    preds_denorm = denormalize_labels(preds, meta)
    labels_denorm = denormalize_labels(labels, meta)

    return preds, labels, preds_denorm, labels_denorm

def compute_metrics(preds, labels, preds_denorm, labels_denorm):
    # normalized MSE
    mse = F.mse_loss(preds, labels).item()

    # MAE in meters (denormalized)
    mae = torch.abs(preds_denorm - labels_denorm).mean().item()

    # Euclidean error (per event)
    errors = torch.sqrt(((preds_denorm - labels_denorm)**2).sum(dim=1))

    return mse, mae, errors

def plot_avg_error_map(preds_denorm, labels_denorm):
    import numpy as np

    x = labels_denorm[:, 0].numpy()
    y = labels_denorm[:, 1].numpy()
    errors = np.sqrt(((preds_denorm - labels_denorm).numpy()**2).sum(axis=1))

    counts, xedges, yedges = np.histogram2d(x, y, bins=50)
    err_sum, _, _ = np.histogram2d(x, y, bins=50, weights=errors)

    avg_error = err_sum / (counts + 1e-6)

    plt.figure(figsize=(6,5))
    plt.imshow(avg_error.T, origin='lower', aspect='auto')
    plt.colorbar(label="Average error")

    plt.xlabel("x bin")
    plt.ylabel("y bin")
    plt.savefig("avg_error_map.png",dpi=500)

    plt.show()
def plot_error_hist(errors):
    plt.figure()
    plt.hist(errors.numpy(), bins=50)

    plt.xlabel("Distance residual (m)")
    plt.ylabel("Count")
    plt.savefig("error_histogram.png",dpi=500)
    plt.show()
    
import matplotlib.pyplot as plt

def plot_2d_residual(preds_denorm, labels_denorm):
    residuals = (preds_denorm - labels_denorm).numpy()
    dx = residuals[:, 0]
    dy = residuals[:, 1]
    l = 2.5

    plt.figure(figsize=(6,6))
    plt.hist2d(dx, dy, bins=120)
    plt.colorbar(label="Counts")

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim(-l, l)
    plt.ylim(-l, l)

    plt.savefig("2d_residual.png",dpi=500)
    plt.show()

##### Code starts here #####

test_dataset = awkward.from_parquet(f'{DATA_PATH}/test.pq')
train_dataset = awkward.from_parquet(f'{DATA_PATH}/train.pq')
val_dataset = awkward.from_parquet(f'{DATA_PATH}/val.pq')

# --- collect global t range ---
t_min = np.inf
t_max = -np.inf

for k in [0, 1, 2, 3, 4]:
    temp = test_dataset[k]['data'].to_numpy()
    t = temp[0, :]
    t_min = min(t_min, t.min())
    t_max = max(t_max, t.max())

meta = norm_help(train_dataset) # meta parameters for z scaling, calculated on the training set
train_dataset_norm = normalize_dataset(train_dataset, meta)
val_dataset_norm   = normalize_dataset(val_dataset, meta)
test_dataset_norm  = normalize_dataset(test_dataset, meta)

train_loader = DataLoader(train_dataset_norm, batch_size=64, shuffle=True, collate_fn=collate_fn_transformer)
val_loader   = DataLoader(val_dataset_norm,   batch_size=64, shuffle=False, collate_fn=collate_fn_transformer)
test_loader  = DataLoader(test_dataset_norm,  batch_size=64, shuffle=False, collate_fn=collate_fn_transformer)

model = TransformerEncoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_losses, val_losses = train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=50,
    lr=1e-3
)
    
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.savefig("loss_curve.png",dpi=500)
plt.show()


preds, labels, preds_denorm, labels_denorm = evaluate_model(
    model, test_loader, device, meta
)

mse, mae, errors = compute_metrics(
    preds, labels, preds_denorm, labels_denorm
)

print(f"MSE (normalized): {mse:.4f}")
print(f"MAE (meters): {mae:.4f}")

plot_error_hist(errors)
plot_2d_residual(preds_denorm, labels_denorm)
plot_avg_error_map(preds_denorm, labels_denorm)