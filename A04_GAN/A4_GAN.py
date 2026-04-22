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
import torchvision.utils as vutils
# FOR TENSOR BOARD VISUALIZATION
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard
import os
import time

ROOT = "/home/andre/courses/AdvMl_V26/A04_GAN"
os.chdir(ROOT)


writer = SummaryWriter(f"runs/GAN_MNIST_{int(time.time())}")

# Hyperparameters
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
batchSize = 32  # Batch size
numEpochs = 100
logStep = 625  # the number of steps to log the images and losses to tensorboard

latent_dimension = 128 # 64, 128, 256
# for simplicity we will flatten the image to a vector and to use simple MLP networks
# 28 * 28 * 1 flattens to 784
# you are also free to use CNNs
image_dimension = 28 * 28 * 1  # 784

# we define a tranform that converts the image to tensor and normalizes it with mean and std of 0.5
# which will convert the image range from [0, 1] to [-1, 1]
myTransforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

# the MNIST dataset is available through torchvision.datasets
print("loading MNIST digits dataset")
dataset = datasets.MNIST(root="dataset/", transform=myTransforms, download=True)
# let's create a dataloader to load the data in batches
loader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

class Generator(nn.Module):
    """
    Generator Model
    """
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(latent_dimension, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, image_dimension),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)



class Discriminator(nn.Module):
    """
    Discriminator Model
    """
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(image_dimension, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


# initialize networks and optimizers
discriminator = Discriminator().to(device)
generator = Generator().to(device)
opt_discriminator = optim.Adam(discriminator.parameters(), lr=lr)
opt_generator = optim.Adam(generator.parameters(), lr=lr)

# This is a binary classification task, so we use Binary Cross Entropy Loss
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, latent_dimension).to(device)

# Training Loop
step = 0
print("Started Training and visualization...")
for epoch in range(numEpochs):
    # loop over batches
    print()
    for batch_idx, (real, _) in enumerate(loader):
        # First we train the discriminator on real images vs. generated images

        # Get the real images and flatten them
        # for simplicity, we flatten the image to a vector and to use simple MLP networks
        # 28 * 28 * 1 flattens to 784
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # Step 1) generate fake images
        noise = torch.randn(batch_size, latent_dimension).to(device)
        fake = generator(noise)

        # Step 2) Train Discriminator:
        # - predict the discriminator output for real images
        disc_real = discriminator(real).view(-1)
        labels_real = torch.ones_like(disc_real)
        loss_real = criterion(disc_real, labels_real)

        # - predict the discriminator output for fake images
        disc_fake = discriminator(fake.detach()).view(-1)
        labels_fake = torch.zeros_like(disc_fake)
        loss_fake = criterion(disc_fake, labels_fake)

        # -average the loss for real and fake images
        loss_discriminator = (loss_real + loss_fake) / 2

        # - update discriminator
        opt_discriminator.zero_grad()
        loss_discriminator.backward(retain_graph=True)
        opt_discriminator.step()


        # Train Generator:
        # - pass the fake images through the discriminator
        output = discriminator(fake).view(-1)

        # - trick: label fake as real (1s)
        labels_gen = torch.ones_like(output)

        # - calculate loss
        loss_generator = criterion(output, labels_gen)

        # - update generator
        opt_generator.zero_grad()
        loss_generator.backward()
        opt_generator.step()


        # print the progress
        print(f"\rEpoch [{epoch}/{numEpochs}] Batch {batch_idx}/{len(loader)} \ Loss discriminator: {loss_discriminator:.4f}, loss generator: {loss_generator:.4f}", end="")

        # Log the losses and example images to tensorboard
        if batch_idx % logStep == 0:
            with torch.no_grad():
                # Generate noise via Generator, we always use the same noise to see the progression
                fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                        # Save image grid
                vutils.save_image(
                    fake,
                    f"images/fake_step_{step}.png",
                    normalize=True,
                    nrow=8
                )
                
                # Get real data
                data = real.reshape(-1, 1, 28, 28)
                # make grid of pictures and add to tensorboard
                imgGridFake = torchvision.utils.make_grid(fake, normalize=True)
                imgGridReal = torchvision.utils.make_grid(data, normalize=True)

                writer.add_image("Fake Images", imgGridFake, global_step=step)
                writer.add_image("Real Images", imgGridReal, global_step=step)

                writer.add_scalar("Loss/Discriminator", loss_discriminator.item(), step)
                writer.add_scalar("Loss/Generator", loss_generator.item(), step)

                # increment step
                step += 1