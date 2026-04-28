import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns  # a useful plotting library on top of matplotlib
from tqdm.auto import tqdm # a nice progress bar

# # This is a simple example of a diffusion model in 1D.


# generate a dataset of 1D data from a mixture of two Gaussians
# this is a simple example, but you can use any distribution
data_distribution = torch.distributions.mixture_same_family.MixtureSameFamily(
    torch.distributions.Categorical(torch.tensor([1, 2])),
    torch.distributions.Normal(torch.tensor([-4., 4.]), torch.tensor([1., 1.]))
)

dataset = data_distribution.sample(torch.Size([10000]))  # create training data set
dataset_validation = data_distribution.sample(torch.Size([1000])) # create validation data set
# fig, ax = plt.subplots(1, 1)
# sns.histplot(dataset)
# plt.show()

# we will keep these parameters fixed throughout
# these parameters should give you an acceptable result
# but feel free to play with them
TIME_STEPS = 250
BETA = torch.tensor(0.02)
N_EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 0.8e-4

betas = torch.full((TIME_STEPS,), BETA)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0) # cumulative product



# define the neural network that predicts the amount of noise that was
# added to the data
# the network should have two inputs (the current data and the time step)
# and one output (the predicted noise)

class NoisePredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x, t):
        t = t.float() / TIME_STEPS
        t = t.unsqueeze(1)
        x = x.unsqueeze(1)
        inp = torch.cat([x, t], dim=1)
        return self.net(inp).squeeze(1)

g = NoisePredictor()
optimizer = torch.optim.Adam(g.parameters(), lr=LEARNING_RATE)


train_losses = []
val_losses = []
epochs = tqdm(range(N_EPOCHS))  # this makes a nice progress bar
for e in epochs: # loop over epochs
    g.train()
    # loop through batches of the dataset, reshuffling it each epoch
    indices = torch.randperm(dataset.shape[0])
    shuffled_dataset = dataset[indices]
    train_loss = 0
    for i in range(0, shuffled_dataset.shape[0] - BATCH_SIZE, BATCH_SIZE):
        # sample a batch of data
        x0 = shuffled_dataset[i:i + BATCH_SIZE]
        
        # sample random timestep
        t = torch.randint(0, TIME_STEPS, (BATCH_SIZE,))
        
        alpha_bar_t = alpha_bars[t]
        noise = torch.randn_like(x0)
        
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        
        pred_noise = g(xt, t)
        
        loss = torch.nn.functional.mse_loss(pred_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss / BATCH_SIZE)

    # compute the loss on the validation set
    g.eval()
    with torch.no_grad():
        x0 = dataset_validation
        t = torch.randint(0, TIME_STEPS, (x0.shape[0],))
        alpha_bar_t = alpha_bars[t]
        noise = torch.randn_like(x0)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

        pred_noise = g(xt, t)
        val_loss = torch.nn.functional.mse_loss(pred_noise, noise)
    val_losses.append(val_loss.item())
    epochs.set_description(f"loss={loss.item():.4f}, val={val_loss.item():.4f}")



def sample_reverse(g, count):
    """
    Sample from the model by applying the reverse diffusion process

    Here, implement algorithm 2 of the DDPM paper (https://arxiv.org/abs/2006.11239)

    Parameters
    ----------
    g : torch.nn.Module
        The neural network that predicts the noise added to the data
    count : int
        The number of samples to generate in parallel

    Returns
    -------
    x : torch.Tensor
        The final sample from the model
    """
    g.eval()
    
    with torch.no_grad():
        x = torch.randn(count)  # start from pure noise
        for t in reversed(range(TIME_STEPS)):
            t_batch = torch.full((count,),t)
            
            alpha = alphas[t]
            alpha_bar = alpha_bars[t]
            
            pred_noise = g(x, t_batch)
            
            coef1 = 1 / torch.sqrt(alpha)
            coef2 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            
            x = coef1 * (x - coef2 * pred_noise)
            
            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(betas[t])
                x += sigma * noise
                
    return x




samples = sample_reverse(g, 1000)
samples = samples.detach().numpy()

# plot the samples
fig, ax = plt.subplots(1, 1)
bins = np.linspace(-10, 10, 50)
sns.kdeplot(dataset, ax=ax, color='blue', label='True distribution', linewidth=2)
sns.histplot(samples, ax=ax, bins=bins, color='red', label='Sampled distribution', stat='density')
ax.legend()
ax.set_xlabel('Sample value')
ax.set_ylabel('Sample count')
