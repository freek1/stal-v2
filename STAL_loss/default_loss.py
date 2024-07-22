import torch

class DefaultLoss(torch.nn.Module):
    """
    (Novel) Encoder Loss function L = L_MI + L_S
    Mutual Information part + Sparsity part.
    """
    def __init__(self):
        super(DefaultLoss, self).__init__()
        self.__name__ = "Default Loss"
        print(f"{self.__name__} initialized.")

    def forward(self, h, X, Z1, Z2):
        """
        W: torch.Tensor[batch, omega, psi], encoded 'spiketrain', not binarized yet (\hat{B} in the paper)
        X: torch.Tensor[batch, omega, psi], input signal (\hat{X} in the paper)
        Z1: torch.Tensor[batch, omega, psi], representation of Layer 1
        Z2: torch.Tensor[batch, omega, psi], representation of Layer 2
        """
        assert isinstance(X, torch.Tensor) and isinstance(h, torch.Tensor), "X and h must be torch tensors."
        
        if Z1 is not None:
            assert X.shape == Z1.shape, f"X and Z1 must have the same shape: {X.shape} != {Z1.shape}."
        if Z2 is not None:
            assert X.shape == Z2.shape, f"X and Z2 must have the same shape: {X.shape} != {Z2.shape}."
        
        n_samples = X.shape[0]
        n_timesteps = X.shape[1]
        h_unroll = h.reshape(n_samples, n_timesteps, -1)
        n_spikes_per_timestep = h_unroll.shape[2]
        weights = torch.arange(1, n_spikes_per_timestep + 1, dtype=h_unroll.dtype, device=h_unroll.device)
        # Reverse the weights, so that the first spike is the most important
        weights = torch.flip(weights, [0])
        h_weighted = torch.sum(h_unroll * weights, dim=2)
        
        mi = compute_mutual_information(X, h_weighted)
        
        mi_Z1 = 0 
        mi_Z2 = 0
        if Z1 is not None:
            mi_Z1 = compute_mutual_information(X, Z1)
        if Z2 is not None:            
            mi_Z2 = compute_mutual_information(X, Z2)
        
        # Instead of punishing many spikes, i.e. every spike (L1 norm),
        # we can try punishing the difference in spikes from the area of the input signal
        
        area_X = torch.sum(X, dim=1)
        n_spikes = torch.sum(h, dim=1)
        
        # E.g., we want to have 525 spikes if the area of X is 525,
        # so we add a loss of the L1 or L2 distance:  
        L1 = torch.abs(n_spikes - (area_X * n_spikes_per_timestep))        
       
        MI = mi
        cnt = 1
        # if Z1 is not None and Z2 is None:
        #     MI += mi_Z1
        #     cnt += 1
        # if Z2 is not None:
        #     MI += mi_Z2
        #     cnt += 2
        loss = -(MI/cnt) + torch.mean(L1) 

        return loss
    
def compute_mutual_information(X, Z):
    """ 
    Computes the mutual information between two random variables X and Z.
    All computations are done using torch operations, to keep the gradient flow.
    """
    if X.shape[0] != Z.shape[0]:
        raise ValueError("X and W must have the same number of samples.")
    
    if X.ndim == 3:
        if X.size(2) == 1:
            X = X.squeeze(2)
    
    eps = torch.tensor(1e-12, dtype=torch.float32)
    joint_prob = torch.mean(torch.multiply(X, Z))
    px = torch.mean(X) # Marginal probability of X
    # px = px if px > 0 else eps
    assert px >= 0, f"Marginal prob. of X is smaller than 0, px: {px}"
    pz = torch.mean(Z) # Marginal probability of Z
    assert pz >= 0, f"Marginal prob. of Z is smaller than 0, pz: {pz}"
    
    if joint_prob == 0 or px == 0 or pz == 0:
        joint_prob = torch.max(joint_prob, eps)
        px = torch.max(px, eps)
        pz = torch.max(pz, eps)
        
    mutual_information = joint_prob * torch.log2((joint_prob) / (px * pz))
    return mutual_information
