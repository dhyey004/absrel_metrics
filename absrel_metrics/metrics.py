import torch

def compute_abs_rel(pred: torch.Tensor, target: torch.Tensor, mask: float = 0.0) -> torch.Tensor:
    """
    Computes the Absolute Relative Error between predicted and target depth maps.
    
    Args:
        pred (torch.Tensor): The predicted depth map.
        target (torch.Tensor): The ground truth depth map.
        mask (float): The value in the target indicating invalid/missing data.
    """
    
    valid_mask = target > mask
    
    
    valid_pred = pred[valid_mask]
    valid_target = target[valid_mask]
    
    # Returns 0 if no valid pixels at all, to prevent NaN error
    if valid_target.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    
    
    absolute_error = torch.abs(valid_pred - valid_target)
    relative_error = absolute_error / valid_target
    
    
    return torch.mean(relative_error) # final value here is a scalar tensor containing mean AbsRel error