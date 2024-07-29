import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import time

from snntorch import spikegen

from STAL.LAST import SpikeThresholdAdaptiveLearning
from STAL_loss.only_mi_loss import MILoss as DefaultLoss

def load_data_roshambo():
    # Specify the folder path
    X = np.load("roshambo/X_roshambo.npy")
    y = np.load("roshambo/y_roshambo.npy")
    
    X = np.array(X, dtype=np.float32)
    print(X.min(), X.max()) # -128 to 127
    X = np.abs(X)
    X = X / X.max()
            
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y-1, dtype=torch.long)

device = torch.device("mps")

X, y = load_data_roshambo()
train_X, test_X, train_y, test_y  = train_test_split(X, y, test_size=0.2, random_state=42)
train_X, val_X, train_y, val_y  = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

n_samples, omega, c = train_X.size()

psi = 50
l1_sz = 0 #omega // 2
l2_sz = 0 #omega // 2
drop_p = 0.

STAL = SpikeThresholdAdaptiveLearning(omega, psi, c, l1_sz, l2_sz, drop_p)
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

print(pixels, spikes)

colors = plt.cm.viridis(np.linspace(0, 1, c))

fig, ax = plt.subplots(2, 1, figsize=(10, 5)) 
for channel in range(c):
    ax[0].plot(x[0, :, channel].numpy(), c=colors[channel], label=f"Channel {channel}")
ax[0].legend()
ax[0].set_title("Input data")
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Amplitude")

ax[1].scatter(pixels, spikes, c=colors[0], label="Spikes channel 0")
ax[1].legend()

plt.suptitle("Random Initialisation")

plt.tight_layout()
plt.savefig("roshambo/roshambo_random.png")
plt.close()

# Train STAL
# --------------------------------

default_loss = DefaultLoss()
optimizer = torch.optim.AdamW(STAL.parameters(), lr=0.01)

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
        x_train = train_X[i:i+batch_size].to(device)
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
            x_val = val_X[i:i+batch_size].to(device)
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
    x_test = test_X[:batch_size].to(device)
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

ax[1].scatter(pixels, spikes, c=colors[0], label="Spikes channel 0")
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

print(B_rate_tr.shape, B_STAL_tr.shape)

b = B_rate_ts[0]
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

ax[1].scatter(pixels, spikes, c=colors[0], label="Spikes channel 0")
ax[1].legend()

plt.suptitle("Rate Coding")

plt.tight_layout()
plt.savefig("roshambo/roshambo_rate.png")
plt.close()
