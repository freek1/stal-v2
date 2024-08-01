import torch

class MILoss(torch.nn.Module):
    """
    (Novel) Encoder Loss function L = L_MI
    Maximize Mutual Information.
    """
    def __init__(self):
        super(MILoss, self).__init__()
        self.__name__ = "Default Loss"
        print(f"{self.__name__} initialized.")

    def forward(self, h, X, Z1, Z2):
        """
        h: torch.Tensor[batch, omega * c * psi], encoded 'spiketrain', not binarized yet (\hat{B} in the paper)
        X: torch.Tensor[batch, omega, c, psi], input signal (\hat{X} in the paper)
        Z1: torch.Tensor[batch, omega * c * psi], representation of Layer 1
        Z2: torch.Tensor[batch, omega * c * psi], representation of Layer 2
        """
        assert isinstance(X, torch.Tensor) and isinstance(h, torch.Tensor), "X and h must be torch tensors."
        
        n_samples = X.shape[0]
        n_timesteps = X.shape[1]
        n_channels = X.shape[2]
        
        h_unroll = h.reshape(n_samples, n_timesteps, n_channels, -1)
        
        n_spikes_per_timestep = h_unroll.shape[3]
        weights = torch.arange(1, n_spikes_per_timestep + 1, dtype=h_unroll.dtype, device=h_unroll.device) * 100
        # Reverse the weights, so that the first spike is the most important
        weights = torch.flip(weights, [0])
        h_weighted = torch.sum(h_unroll * weights, dim=3)
        
        mi = compute_mutual_information(X, h_weighted)
       
        MI = mi
        loss = -MI

        return loss
    
def compute_mutual_information(X, Y):
    """ 
    Computes the mutual information between two random variables X and Y.
    All computations are done using torch operations, to keep the gradient flow.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")
    
    if X.ndim == 3:
        if X.size(2) == 1:
            X = X.squeeze(2)
    
    eps = torch.tensor(1e-12, dtype=torch.float32)
    joint_prob = torch.mean(torch.multiply(X, Y))
    px = torch.mean(X) # Marginal probability of X
    # px = px if px > 0 else eps
    assert px >= 0, f"Marginal prob. of X is smaller than 0, px: {px}"
    py = torch.mean(Y) # Marginal probability of Z
    assert py >= 0, f"Marginal prob. of Y is smaller than 0, pz: {py}"
    
    if joint_prob == 0 or px == 0 or py == 0:
        joint_prob = torch.max(joint_prob, eps)
        px = torch.max(px, eps)
        py = torch.max(py, eps)
        
    mutual_information = joint_prob * torch.log2((joint_prob) / (px * py))
    return mutual_information
