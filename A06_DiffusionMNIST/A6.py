import torchvision
# For image transforms
from torchvision import transforms
# For DATA SET
import torchvision.datasets as datasets
# For Pytorch methods
import torch
import torch.nn as nn
# For Optimizer
import torch.optim as optim
# FOR DATA LOADER
from torch.utils.data import DataLoader

from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import matplotlib.pyplot as plt

import torchvision.utils as vutils
import os
from torch.utils.data import Subset

os.makedirs("samples", exist_ok=True)

def save_grid(images, filename):
    imgs = images.detach().cpu()

    grid = vutils.make_grid(imgs, nrow=4)
    vutils.save_image(grid, filename)


# Hyperparameters
LEARNING_RATE = 4e-4 #4e-4 
BATCH_SIZE = 128 # 128  # Batch size
N_EPOCHS = 30 # 100
IMAGE_SIZE = 28 # 28
TIME_STEPS = 1000 # 1000
SAMPLING_TIMESTEPS = 250 # 259

# we define a tranform that converts the image to tensor
myTransforms = transforms.Compose([transforms.ToTensor()]) # This already normalizes the pixel values to [0, 1] range, which is suitable for our model.

# the MNIST dataset is available through torchvision.datasets
print("loading MNIST digits dataset")
dataset = datasets.MNIST(root="dataset/", transform=myTransforms, download=True)

subset = Subset(dataset,  range(10000))  # Use only the first 10,000 samples
dataset = subset
# let's create a dataloader to load the data in batches
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, download=False, transform=myTransforms)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



DIM = 32
DIM_MULTS = (1, 2, 5)
model = Unet(
    dim = DIM,
    dim_mults = DIM_MULTS,
    flash_attn = False,
    channels = 1
)

diffusion = GaussianDiffusion(
    model,
    image_size = IMAGE_SIZE,
    timesteps = TIME_STEPS,           # number of steps
    sampling_timesteps = SAMPLING_TIMESTEPS    # number of sampling timesteps (using ddim for faster inference [see ddim paper])
)

optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
diffusion = diffusion.to(device)

train_losses = []
test_losses = []
for epoch in range(N_EPOCHS):
    model.train()
    running_loss = 0.0

    for images, _ in loader:
        images = images.to(device)

        loss = diffusion(images)

        optim.zero_grad()
        loss.backward()
        optim.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    train_losses.append(avg_loss)
    
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            loss = diffusion(images)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    print(f"Epoch {epoch+1}, Train: {avg_loss:.4f}, Val: {test_loss:.4f}")


    # ---- SAMPLE EACH EPOCH ----
    with torch.no_grad():
        samples = diffusion.sample(batch_size=16)

    save_grid(samples, f"samples/epoch_{epoch+1}.png")

model.eval()
with torch.no_grad():
    final_samples = diffusion.sample(batch_size=16)

save_grid(final_samples, "samples/final.png")


plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.grid()
plt.savefig("loss_plot.png")
plt.show()