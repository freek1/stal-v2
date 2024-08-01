import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import time

from snntorch import spikegen

from STAL.LAST import LearningAdaptiveSpikeThresholds
from STAL_loss.only_mi_loss import MILoss as DefaultLoss

def load_data_roshambo():
    # Specify the folder path
    X = np.load("roshambo/X_roshambo.npy")
    y = np.load("roshambo/y_roshambo.npy")
    
    X = np.array(X, dtype=np.float32)
    X = np.abs(X)
    X = X / X.max()
            
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y-1, dtype=torch.long)

device = torch.device("mps")

X, y = load_data_roshambo()
train_X, test_X, train_y, test_y  = train_test_split(X, y, test_size=0.2, random_state=42)
train_X, val_X, train_y, val_y  = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

print(train_X.size(0), val_X.size(0), test_X.size(0))

n_samples, omega, c = train_X.size()

psi = 50
l1_sz = 100 #omega // 2
l2_sz = 100 #omega // 2
drop_p = 0.

STAL = LearningAdaptiveSpikeThresholds(omega, psi, c, l1_sz, l2_sz, drop_p)
STAL.print_learnable_params()

batch_size = 64
x = test_X[:batch_size]

h, Z1, Z2 = STAL(x)

theta = 0.99
spiketrain = (h > theta).float()

B = spiketrain.reshape(batch_size, omega, c, psi)
# bsz, omega, c, psi

b = B[0]
# start with 1 channel
b = b[:, 0, :]
pixels, spikes = np.where(b == 1)

colors = plt.cm.viridis(np.linspace(0, 1, c))

fig, ax = plt.subplots(2, 1, figsize=(10, 5)) 
for channel in range(c):
    ax[0].plot(x[0, :, channel].numpy(), c=colors[channel], label=f"Channel {channel}")
ax[0].legend()
ax[0].set_title("Input data")
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Amplitude")

ax[1].scatter(pixels, spikes, color=colors[0], label="Spikes channel 0")
ax[1].set_xlim(0, omega)
ax[1].set_ylabel(r"Spike Time Steps ($\psi$)")
ax[1].legend()

plt.suptitle("Random Initialisation")

plt.tight_layout()
plt.savefig("roshambo/roshambo_random.png")
plt.close()

# Train STAL
# --------------------------------

default_loss = DefaultLoss()
optimizer = torch.optim.AdamW(STAL.parameters(), lr=0.1)

n_epochs = 20
STAL.to(device)

start = time.time()

train_loss = []
train_spiketrains = []
val_loss = []
val_spiketrains = []
for epoch in range(n_epochs):
    e = []
    for i in range(0, len(train_X), batch_size):
        bsz = max(0, min(batch_size, len(test_X) - i))
        if bsz == 0:
            break
        x_train = train_X[i:i+bsz].to(device)
        optimizer.zero_grad()
        h, Z1, Z2 = STAL(x_train)
        if epoch == n_epochs - 1:
            spiketrain = (h > theta).float()
            train_spiketrains = np.append(train_spiketrains, spiketrain.detach().cpu().numpy())
        loss = default_loss(h, x_train, Z1, Z2)
        loss.backward()
        optimizer.step()
        e.append(loss.item())
    train_loss.append(np.mean(e))
    print(f"Epoch {epoch + 1} - Train Loss: {train_loss[-1]:.3f}")
    
    e = []
    with torch.no_grad():
        for i in range(0, len(val_X), batch_size):
            bsz = max(0, min(batch_size, len(test_X) - i))
            if bsz == 0:
                break
            x_val = val_X[i:i+bsz].to(device)
            h, Z1, Z2 = STAL(x_val)
            if epoch == n_epochs - 1:
                spiketrain = (h > theta).float()
                val_spiketrains = np.append(val_spiketrains, spiketrain.detach().cpu().numpy())
            loss = default_loss(h, x_val, Z1, Z2)
            e.append(loss.item())
    val_loss.append(np.mean(e))
    print(f"\t- Validation Loss: {val_loss[-1]:.3f}")

# Test
test_loss = []
test_spiketrains = []
for i in range(0, len(test_X), batch_size):
    bsz = max(0, min(batch_size, len(test_X) - i))
    if bsz == 0:
        break
    x_test = test_X[:bsz].to(device)
    h, Z1, Z2 = STAL(x_test)
    spiketrain = (h > theta).float()
    test_spiketrains = np.append(test_spiketrains, spiketrain.detach().cpu().numpy())
    loss = default_loss(h, x_test, Z1, Z2)
    test_loss.append(loss.item())

print("---")
print(f"Test Loss: {np.mean(test_loss):.3f}")
print("---")

end = time.time()
print(f"Training took {end-start:.2f}s")

def plot_loss(train_loss, val_loss, test_loss):
    plt.plot(train_loss, label="Train")
    plt.plot(val_loss, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.hlines(np.mean(test_loss), 0, len(train_loss) - 1, colors='black', linestyles='dashed', label="Test")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roshambo/loss_curves.png")
    plt.close()

plot_loss(train_loss, val_loss, test_loss)

# Inspect the spike train after training
STAL.cpu()
STAL.eval()

B_STAL_tr = train_spiketrains.reshape(-1, omega, c, psi)
B_STAL_val = val_spiketrains.reshape(-1, omega, c, psi)
B_STAL_ts = test_spiketrains.reshape(-1, omega, c, psi)

# ------------
# Inspect the spike train after training
# ------------

b = B_STAL_ts[0]
# bsz, omega, c, psi

# start with 1 channel
b = b[:, 0, :]
pixels, spikes = np.where(b == 1)

colors = plt.cm.viridis(np.linspace(0, 1, c))

fig, ax = plt.subplots(2, 1, figsize=(10, 5)) 
for channel in range(c):
    ax[0].plot(x[0, :, channel].numpy(), c=colors[channel], label=f"Channel {channel}")
ax[0].legend()
ax[0].set_title("Input data")
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Amplitude")

ax[1].scatter(pixels, spikes, color=colors[0], label="Spikes channel 0")
ax[1].set_xlim(0, omega)
ax[1].set_ylabel(r"Spike Time Steps ($\psi$)")
ax[1].legend()

plt.suptitle("Post Training")

plt.tight_layout()
plt.savefig("roshambo/roshambo_trained.png")
plt.close()

# Rate encoding
# ------------
# Swap indices to match B_STAL's shape
B_rate_tr = spikegen.rate(train_X, psi).permute(1, 2, 3, 0)
B_rate_val = spikegen.rate(val_X, psi).permute(1, 2, 3, 0)
B_rate_ts = spikegen.rate(test_X, psi).permute(1, 2, 3, 0)

b = B_rate_ts[0]
# bsz, omega, c, psi

# start with 1 channel
b = b[:, 0, :]
pixels, spikes = np.where(b == 1)

colors = plt.cm.viridis(np.linspace(0, 1, c))

fig, ax = plt.subplots(2, 1, figsize=(10, 5)) 
for channel in range(c):
    ax[0].plot(x[0, :, channel].numpy(), color=colors[channel], label=f"Channel {channel}")
ax[0].legend()
ax[0].set_title("Input data")
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Amplitude")

ax[1].scatter(pixels, spikes, color=colors[0], label="Spikes channel 0")
ax[1].set_xlim(0, omega)
ax[1].set_ylabel(r"Spike Time Steps ($\psi$)")
ax[1].legend()

plt.suptitle("Rate Coding")

plt.tight_layout()
plt.savefig("roshambo/roshambo_rate.png")
plt.close()

## Compare the spike trains
# - Mutual Information
# - Spike Density

# MI
from STAL_loss.only_mi_loss import compute_mutual_information

B_STAL_ts = torch.tensor(B_STAL_ts)

def weight(B):
    n_samples = B.shape[0]
    n_timesteps = B.shape[1]
    n_channels = B.shape[2]
    
    h_unroll = B.reshape(n_samples, n_timesteps, n_channels, -1)
    
    n_spikes_per_timestep = h_unroll.shape[3]
    weights = torch.arange(1, n_spikes_per_timestep + 1, dtype=h_unroll.dtype, device=h_unroll.device) * 10
    # Reverse the weights, so that the first spike is the most important
    weights = torch.flip(weights, [0])
    h_weighted = torch.sum(h_unroll * weights, dim=3)
    
    return h_weighted

MI_STAL_ts = compute_mutual_information(test_X, weight(B_STAL_ts))
MI_rate_ts = compute_mutual_information(test_X, weight(B_rate_ts))

print(f"MI STAL: {MI_STAL_ts:.3f}")
print(f"MI Rate: {MI_rate_ts:.3f}")

def spike_density(B):
    n_spikes = B.sum(dim=3) / B.shape[3] 
    return n_spikes.mean()

dens_STAL_ts = spike_density(B_STAL_ts)
dens_rate_ts = spike_density(B_rate_ts)

print(f"Spike Density STAL: {dens_STAL_ts:.3f}")
print(f"Spike Density Rate: {dens_rate_ts:.3f}")
