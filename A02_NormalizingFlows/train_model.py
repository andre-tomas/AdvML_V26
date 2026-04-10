import time
import sys
import os
#ROOT = "/home/andre/courses/AdvMl_V26/A02_NormalizingFlows"
#os.chdir(ROOT)
import argparse
import glob
import subprocess
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from matplotlib import pyplot as plt
import jammy_flows
from scipy.stats import norm
from helper import *



DATA_PATH = "/home/andre/courses/AdvMl_V26/A01_Uncertainty/DATA"
fp64_on_cpu = False


# Hyper parameters
batch_size = 64


# Call the function to get normalized data
spectra, labels, spectra_length, n_labels, labelNames, ranges = get_normalized_data(DATA_PATH)


# Define the CNN encoder model. The output of the model is the input to the normalizing flow.
# The latent dimension is the number of parameters in the normalizing flow.
class TinyCNNEncoder(nn.Module):
    def __init__(self, input_dim,latent_dimension):
        
        Nc = 16
        kernel_size = 6
        stride = 2
        super(TinyCNNEncoder, self).__init__()
    

        self.model = nn.Sequential(
            nn.Conv1d(1, Nc, 4*kernel_size, 4*stride),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(Nc, 2*Nc, kernel_size, stride),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(2*Nc, 4*Nc, kernel_size, stride),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(4*Nc, 8*Nc, kernel_size, stride),
            nn.ReLU(),

            nn.Flatten(),

            nn.LazyLinear(2048),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.LazyLinear(1024),
            nn.ReLU(),


            nn.LazyLinear(latent_dimension),
        )

    def forward(self, x):
        x = self.model(x)
        return x


def nf_loss(inputs, batch_labels, model):
    """
    Computes the loss for a normalizing flow model.

    Parameters
    ----------
    inputs : torch.Tensor
        The input data to the model.
    batch_labels : torch.Tensor
        The labels corresponding to the input data.
    model : torch.nn.Module
        The normalizing flow model used for evaluation.
    Returns
    -------
    torch.Tensor
        The computed loss value.
    """
    log_pdfs = model.log_pdf_evaluation(batch_labels, inputs) # get the probability of the labels given the input data
    loss = -log_pdfs.mean() # take the negative mean of the log probabilities
    return loss


# Defining the normalizng flow model is a bit more involved and requires knowledge of the jammy_flows library.
# Therefore, we provide the relevant code here.
class CombinedModel(nn.Module):
    """
    A combined model that integrates a normalizing flow with a CNN encoder.
    """

    def __init__(self, encoder, nf_type="diagonal_gaussian"):
        """
        Initializes the normalizing flow model.

        Parameters
        ----------
        encoder : callable
            A function or callable object that returns an encoder model. The encoder model
            should take the number of flow parameters as input and output the latent dimension.
        nf_type : str, optional
            The type of normalizing flow to use. Options are "diagonal_gaussian", "full_gaussian",
            and "full_flow". Default is "diagonal_gaussian".
        Raises
        ------
        Exception
            If an unknown `nf_type` is provided.
        Notes
        -----
        This method sets up a 3-dimensional probability density function (PDF) over Euclidean space (e3)
        using the specified normalizing flow type. The flow structure and options are configured based on
        the provided `nf_type`. The PDF is created using the `jammy_flows` library, and the number of flow
        parameters is determined and printed. The encoder is initialized with the number of flow parameters.
        """

        super().__init__()

        # we define a 3-d PDF over Euclidean spae (e3)
        # using recommended settings (https://github.com/thoglu/jammy_flows/issues/5 scroll down)
        opt_dict = {}
        opt_dict["t"] = {}
        if (nf_type == "diagonal_gaussian"):
            opt_dict["t"]["cov_type"] = "diagonal"
            flow_defs = "t"
        elif (nf_type == "full_gaussian"):
            opt_dict["t"]["cov_type"] = "full"
            flow_defs = "t"
        elif (nf_type == "full_flow"):
            opt_dict["t"]["cov_type"] = "full"
            flow_defs = "gggt"
        else:
            raise Exception("Unknown nf type ", nf_type)

        opt_dict["g"] = dict()
        opt_dict["g"]["fit_normalization"] = 1
        opt_dict["g"]["upper_bound_for_widths"] = 1.0
        opt_dict["g"]["lower_bound_for_widths"] = 0.01

        self.nf_type = nf_type

        # 3d PDF (e3) with ggggt flow structure. Four Gaussianation-flow (https://arxiv.org/abs/2003.01941) layers ("g") and an affine flow ("t")
        self.pdf = jammy_flows.pdf("e3", flow_defs, options_overwrite=opt_dict,
                                   amortize_everything=True, amortization_mlp_use_custom_mode=True)

        # get the number of flow parameters
        num_flow_parameters = self.pdf.total_number_amortizable_params

        print("The normalizing flow has ", num_flow_parameters, " parameters...")

        # latent dimension (output of the CNN encoder) is set to 128
        input_dim = spectra.shape[1]
        self.encoder = encoder(input_dim,num_flow_parameters)

    def log_pdf_evaluation(self, target_labels, input_data):
        """
        Evaluate the log probability density function (PDF) for the given target labels and input data.

        The normalizing flow parameters are predicted by the encoder network based on the input data.
        Then, the log PDF is evaluated at the position of the label.

        Parameters:
        -----------
        target_labels : torch.Tensor
            The target labels for which the log PDF is to be evaluated.
        input_data : torch.Tensor
            The input data to be encoded and used for evaluating the log PDF.
        Returns:
        --------
        log_pdf : torch.Tensor
            The evaluated log PDF for the given target labels and input data.
        """
        latent_intermediate = self.encoder(input_data)  # get the flow parameters from the CNN encoder

        if (self.nf_type == "full_flow"):
            # convert to double. Double precision is needed for the Gaussianization flow. This is for numerical stability.
            if fp64_on_cpu:  # MPS does not support double precision, therefore we need to run the flow on the CPU
                latent_intermediate = latent_intermediate.cpu().to(torch.float64)
                target_labels = target_labels.cpu().to(torch.float64)
            else:
                latent_intermediate = latent_intermediate.to(torch.float64)
                target_labels = target_labels.to(torch.float64)

        # evaluate the log PDF at the target labels
        log_pdf, _, _ = self.pdf(target_labels, amortization_parameters=latent_intermediate)
        return log_pdf

    def sample(self, flow_params, samplesize_per_batchitem=1000):
        """
        Sample new points from the PDF given input data.

        Parameters
        ----------
        flow_params : tensor
            Parameters for the normalizing flow, must be of shape (B, L) where B is the batch size and L is the latent dimension.
        samplesize_per_batchitem : int, optional
            Number of samples to draw per batch item. Defaults to 1000.

        Returns
        -------
        tensor
            A tensor of shape (B, S, D) where B is the batch dimension, S is the number of samples, 
            and D is the dimension of the target space for the samples.
        """
        # for full flow we need to convert to double precision for the normalizing flow
        # for numerical stability
        if (self.nf_type == "full_flow"):
            # convert to double
            if fp64_on_cpu: # MPS does not support double precision, therefore we need to run the flow on the CPU
                flow_params = flow_params.cpu().to(torch.float64)
            else:
                flow_params = flow_params.to(torch.float64)

        batch_size = flow_params.shape[0] # get the batch size
        # sample from the normalizing flow
        repeated_samples, _, _, _ = self.pdf.sample(amortization_parameters=flow_params.repeat_interleave(
            samplesize_per_batchitem, dim=0), allow_gradients=False)

        # reshape the samples to be grouped by batch item
        reshaped_samples = repeated_samples[:, None, :].view(
            batch_size, samplesize_per_batchitem, -1)

        return reshaped_samples

    def forward(self, input_data, samplesize_per_batchitem=1000):
        """
        Perform a forward pass through the model, predicting the mean and standard deviation of the samples.

        Normalizing flows do not directly predict the target labels. Instead, they predict the parameters of the flow that
        transforms the base distribution to the target distribution. Often, we still want to predict the target labels.
        Then, we can sample from the distribution and form the mean of the samples and their standard deviations.
        This is what this function does.

        Parameters
        ----------
        input_data : torch.Tensor
            The input data tensor.
        Returns
        -------
        torch.Tensor
            A tensor of size (B, D*2) where the first half (size D) are the means, 
            the second half (another D) are the standard deviations.
        """
        flow_params=self.encoder(input_data)
        samples=self.sample(flow_params, samplesize_per_batchitem=samplesize_per_batchitem)

        # form mean along dim 1 (samples)
        means=samples.mean(dim=1)
        # form std along dim 1 (samples)
        std_deviations=samples.std(dim=1)

        # return means and std deviations as a concatenated tensor along dim 1
        return torch.cat([means, std_deviations], dim=1)

    def visualize_pdf(self, input_data, filename, samplesize=1000, batch_index=0, truth=None):
        """
        Visualizes the probability density function (PDF) of the given input data using a normalizing flow model.

        The function generates samples from the normalizing flow (using the sample() function) 
        and plots the histogram of the samples together with a Gaussian approximation.

        Parameters
        ----------
        input_data : torch.Tensor
            The input data tensor from which to pick one batch item for visualization.
        filename : str
            The filename where the resulting plot will be saved.
        samplesize : int, optional
            The number of samples to generate for the PDF visualization (default is 10000).
        batch_index : int, optional
            The index of the batch item to visualize (default is 0).
        truth : torch.Tensor, optional
            The true values of the labels, used for comparison in the plot (default is None).

        Returns
        -------
        None
        """
        # pick out one input from batch
        input_bitem = input_data[batch_index:batch_index+1]

        # get the flow parameters (by passing the input data through the CNN encoder network)
        flow_params = self.encoder(input_bitem)

        # sample from the normalizing flow (i.e. samples are drawn from the base distribution and transformed by the flow
        # using the change-of-variable formula)
        samples = self.sample(flow_params, samplesize_per_batchitem=samplesize)
        # the rest of the code is just plotting.

        # we only have 1 batch item
        samples = samples.squeeze(0)

        # plot three 1-dimensional distributions together with normal approximation,
        # so we calculate the mean and std of the samples
        mean = samples.mean(dim=0).cpu().numpy()
        std = samples.std(dim=0).cpu().numpy()
        samples = samples.cpu().numpy()

        fig, axdict = plt.subplots(3, 1)
        for dim_ind in range(3):
            # plot the histogram of the samples
            axdict[dim_ind].hist(samples[:, dim_ind], color="k", density=True,
                                 bins=50, alpha=0.5, label="density based on samples")

            # plot the Gaussian approximation
            min_sample = samples[:, dim_ind].min()
            max_sample = samples[:, dim_ind].max()
            xvals = np.linspace(min_sample, max_sample, 1000)
            yvals = norm.pdf(xvals, loc=mean[dim_ind], scale=std[dim_ind])
            axdict[dim_ind].plot(xvals, yvals, color="green",
                                 label="Gaussian approximation")

            # plot the true value if it is given
            if (truth is not None):
                true_value = truth[dim_ind].cpu().item()
                axdict[dim_ind].axvline(
                    true_value, color="red", label="true value")

            # plot the legend only for the first panel
            if (dim_ind == 0):
                axdict[dim_ind].legend()

        plt.savefig(filename)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-normalizing_flow_type", default="diagonal_gaussian",
                        choices=["diagonal_gaussian", "full_gaussian", "full_flow"])
    args = parser.parse_args()
    print("Using normalizing flow type ", args.normalizing_flow_type)

    model = CombinedModel(TinyCNNEncoder, nf_type=args.normalizing_flow_type)

    # Detect and use Apple Silicon GPU (MPS) if available
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if args.normalizing_flow_type == "full_flow" and device.type == "mps":
        # MPS does not support double precision, therefore we need to run the flow on the CPU
        fp64_on_cpu = True
    print(f"Using device: {device}, performing fp64 on CPU: {fp64_on_cpu}")
    model.to(device)
    
    normalizing_flow_type = args.normalizing_flow_type
    print(spectra.shape)
    
    num_spectra = spectra.shape[0]
    
    train_size = int(0.7*num_spectra)
    val_size = int(0.15*num_spectra)
    test_size = num_spectra-train_size-val_size
    p = 0.95
    y_data,ranges = normalize(labels,p)

    x_tensor = torch.tensor(spectra,dtype=torch.float32).view(num_spectra,-1).to(device)
    #x_tensor = nn.MaxPool1d(2)(x_tensor)
    y_tensor = torch.tensor(y_data,dtype=torch.float32).view(num_spectra,-1).to(device)
    train_dataset, val_dataset, test_dataset = random_split(TensorDataset(x_tensor,y_tensor),[train_size,val_size,test_size])

    
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    
    
    
    num_epoch = 30
    # Model initialization
    loss_func = nf_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-6)

    best_val_loss = float("inf")
    best_path = "best_model.pt"

    train_losses, val_losses = [], []

    for epoch in range(num_epoch):
        print(f"Epoch {epoch+1}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # ---- Train ----
        model.train()
        train_loss = 0
        for batch_x, batch_y in tqdm(train_loader):
            batch_x = batch_x.unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_func(batch_x, batch_y, model)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        print(f"Loss {train_loss:.3f}")

        # ---- Validation ----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.unsqueeze(1)
                loss = loss_func(batch_x, batch_y, model)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Val Loss {val_loss:.3f}")

        # ---- Save best ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"✅ Saved new best model (val_loss={val_loss:.4f})")

        scheduler.step()
        
        
    
    all_pred = []
    all_true = []
    # Test data
    model.load_state_dict(torch.load(best_path))
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.unsqueeze(1)
            predictions = model(batch_x)
            all_pred.append(predictions.cpu())  # move to CPU to save GPU memory
            all_true.append(batch_y.cpu())

    # Concatenate all batches into single tensors, then convert to numpy
    all_pred = torch.cat(all_pred, dim=0).numpy()
    all_true = torch.cat(all_true, dim=0).numpy()

    # Rescale back to original unitsd
    all_pred_original = helper.denormalize(all_pred[:,:n_labels],ranges)
    all_true_original = helper.denormalize(all_true,ranges)

    std_pred_original = helper.denormalize_std(all_pred[:,n_labels:],ranges)

    label_names = ["T_eff", "LogG","Metalicity"]

    fig, axs = plt.subplots(1,3,figsize=(12,4))
    alpha = 0.5
    ax_labels = ["a)","b)","c)","d)","e)","f)","g)","h)","i)"]

    for k,ax in enumerate(axs):

        # Calculate data ranges
        ax_max = np.maximum(all_true_original[:,k].max(),all_pred_original[:,k].max())
        ax_min = np.minimum(all_true_original[:,k].min(),all_pred_original[:,k].min())
        x = np.linspace(ax_min, ax_max, 100)
        ax.plot(x,x,color="black",linestyle="--")

        # add R2 to the plot
        ss_res = np.sum((all_true_original[:,k] - all_pred_original[:,k])**2)
        ss_tot = np.sum((all_true_original[:,k] - np.mean(all_true_original[:,k]))**2)
        r2 = 1 - ss_res/ss_tot
        ax.text(0.05, 0.95, f"$R^2$ = {r2:.3f}", transform=ax.transAxes, fontsize=12, verticalalignment='top')
        ax.text(-0.18,0.99, ax_labels[k], transform=ax.transAxes, fontsize=12, verticalalignment='top',fontdict={"weight":"bold"})

        ax.scatter(all_pred_original[:,k],all_true_original[:,k],alpha=alpha,s=1)
        ax.set_xlabel(f"Predicted {label_names[k]}")
        ax.set_ylabel(f"True {label_names[k]}")

        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)

    plt.tight_layout()
    plt.savefig(f"predictions_{normalizing_flow_type}.pdf",dpi=300)
    plt.show()

    model.visualize_pdf(batch_x, f"pdfs_{normalizing_flow_type}.png", truth=batch_y[0])

    plt.figure(figsize=(5,3))
    plt.plot(train_losses,label="Train")
    plt.plot(val_losses,label="Val")
    plt.ylabel("nf Loss")
    plt.xlabel("Epoch")
    plt.ylim(-7,2)
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"loss_{normalizing_flow_type}.pdf",dpi=300)
    
    fig, axs = plt.subplots(3, 3, figsize=(12, 6), sharey='row')

    bins_pull = 60
    bins_res = 60

    for k in range(3):

        # --- Data ---
        residuals_original = all_pred_original[:,k] - all_true_original[:,k]

        std = all_pred[:,k+3]
        pulls = (all_pred[:,k]-all_true[:,k]) / std

        # Top row: pulls
        ax = axs[0, k]

        ax.hist(pulls, bins=bins_pull)
        ax.set_title(f"{label_names[k]}")
        ax.set_xlabel("(Pred - True) / Std")

        ax.text(0.05, 0.95,
                f"μ={np.mean(pulls):.3f}\nσ={np.std(pulls):.3f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top')
        ax.text(-0.18,0.99, ax_labels[k], transform=ax.transAxes, fontsize=12, verticalalignment='top',fontdict={"weight":"bold"})
        ax.set_xlim(-6,6)

        # middle row: residuals
        ax = axs[1, k]

        ax.hist(residuals_original, bins=bins_res)
        ax.set_xlabel("Pred - True")

        ax.text(0.05, 0.95,
                f"μ={np.mean(residuals_original):.3f}\nσ={np.std(residuals_original):.3f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top')
        ax.text(-0.18,0.99, ax_labels[k+3], transform=ax.transAxes, fontsize=12, verticalalignment='top',fontdict={"weight":"bold"})

        ax = axs[2, k]

        ax.hist(std_pred_original[:,k], bins=bins_res)
        ax.set_xlabel("Predicted Std")

        ax.text(-0.18,0.99, ax_labels[k+6], transform=ax.transAxes, fontsize=12, verticalalignment='top',fontdict={"weight":"bold"})


    axs[0,0].set_ylabel("Count")
    axs[1,0].set_ylabel("Count")
    axs[2,0].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"residuals_pulls_{normalizing_flow_type}.pdf", dpi=300)
    plt.show()