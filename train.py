import time
import warnings
import numpy as np
from tqdm import tqdm
from sksurv.metrics import concordance_index_censored

import torch
import torch.optim as optim
from torch.cuda import amp

from dataset import build_loader
from models.snn_model import build_model, build_loss
from configs.survival_config import get_config
from utils import build_scheduler, fix_random_seeds , MetricLogger
 
warnings.filterwarnings('ignore')

# Training settings
config = get_config()
fix_random_seeds(config.SEED) 
    
train_loader,val_loader,test_loader = build_loader(config)

model = build_model(config)

if config.DEVICES == 'cuda':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.cuda()
    print('Device: {}\n'.format(device))
else:
    device = "cpu"

optimizer = optim.AdamW(model.parameters(), lr=config.TRAIN.BASE_LR,
                        betas=config.TRAIN.OPTIMIZER.BETAS, weight_decay=config.TRAIN.WEIGHT_DECAY)
lr_scheduler = build_scheduler(optimizer, config, train_loader)
losses_dict = build_loss()


def prepare_omics_data(omics_data, device):
    """Prepare omics data by converting to tensor and moving to device."""
    gene_data = []
    for i, omic in enumerate(omics_data):
        gene_data.append(omic.to(device, dtype=torch.float))
    return gene_data


def calculate_c_index(risk_scores, censor_status, survival_time):
    """Calculate C-index metric for survival analysis."""
    risk_scores = np.concatenate(risk_scores)
    censor_status = np.concatenate(censor_status)
    survival_time = np.concatenate(survival_time)
    
    # Check for NaN values
    if np.any(np.isnan(risk_scores)):
        raise ValueError("Risk scores contain NaN values")
    
    # Convert to survival analysis format (censor=1 indicates event occurred)
    event_occurred = (1 - censor_status).astype(bool)
    
    c_index = concordance_index_censored(
        event_occurred, 
        survival_time, 
        risk_scores, 
        tied_tol=1e-08
    )[0]
    
    return c_index, risk_scores, censor_status, survival_time


def train_one_epoch(model, train_loader, optimizer, device, losses_dict):
    """
    Train the model for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Computation device (CPU/GPU)
        losses_dict: Dictionary of loss functions
    
    Returns:
        tuple: (average_loss, c_index)
    """
    model.train()
    
    # Initialize statistics
    total_loss = 0.0
    all_risk_scores = []
    all_censor_status = []
    all_survival_time = []
    
    # Hyperparameters (alpha adjusted from 0.7 to 0.4 to match loss function)
    alpha = 0.4
    eps = 1e-7
    
    # Use progress bar
    pbar = tqdm(
        enumerate(train_loader), 
        total=len(train_loader), 
        desc='Training'
    )
    
    for batch_idx, batch_data in pbar:
        # Unpack batch data
        (
            survival_time_batch,
            censor_status_batch,
            labels_batch,
            *omics_batches,
            features_batch
        ) = batch_data
        
        # Prepare data and move to device
        omics_data = prepare_omics_data(omics_batches, device)
        survival_time = survival_time_batch.to(device, dtype=torch.float32)
        censor_status = censor_status_batch.to(device, dtype=torch.float32)
        features = features_batch.to(device, dtype=torch.float32)
        labels = labels_batch.to(device, dtype=torch.long)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions, survival_scores = model.forward(omics_data)
        
        # Calculate loss
        loss = losses_dict["SURCELoss"](
            predictions, 
            survival_scores, 
            labels, 
            censor_status, 
            alpha=alpha, 
            eps=eps
        )
        
        # Backward pass
        loss.backward()
        
        # Optimizer update
        optimizer.step()
        
        # Accumulate statistics
        total_loss += loss.item()
        
        # Calculate risk scores (risk = -sum of survival functions)
        risk_scores = -torch.sum(survival_scores, dim=1)
        
        # Save statistics for evaluation
        all_risk_scores.append(risk_scores.detach().cpu().numpy().reshape(-1))
        all_censor_status.append(censor_status.detach().cpu().numpy().reshape(-1))
        all_survival_time.append(survival_time.detach().cpu().numpy().reshape(-1))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'batch': batch_idx + 1
        })
    
    # Calculate average loss for the epoch
    avg_loss = total_loss / len(train_loader)
    
    # Calculate C-index
    try:
        c_index, _, _, _ = calculate_c_index(
            all_risk_scores, 
            all_censor_status, 
            all_survival_time
        )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print training summary
        print(f"\nEpoch Summary:")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Average Loss:  {avg_loss:.6f}")
        print(f"  C-index:       {c_index:.4f}")
        print("-" * 40)
        
        return avg_loss, c_index
        
    except ValueError as e:
        print(f"\nWarning: {e}")
        return avg_loss, None


@torch.no_grad()
def validate_one_epoch(model, valid_loader, device):
    """
    Validate the model for one epoch.
    
    Args:
        model: Model to validate
        valid_loader: Validation data loader
        device: Computation device (CPU/GPU)
    
    Returns:
        float: Validation C-index
    """
    model.eval()
    
    # Initialize statistics
    all_risk_scores = []
    all_censor_status = []
    all_survival_time = []
    
    # Use progress bar
    pbar = tqdm(
        enumerate(valid_loader), 
        total=len(valid_loader), 
        desc='Validating'
    )
    
    for batch_idx, batch_data in pbar:
        # Unpack batch data
        (
            survival_time_batch,
            censor_status_batch,
            labels_batch,
            *omics_batches,
            features_batch
        ) = batch_data
        
        # Prepare data and move to device
        omics_data = prepare_omics_data(omics_batches, device)
        survival_time = survival_time_batch.to(device, dtype=torch.float32)
        censor_status = censor_status_batch.to(device, dtype=torch.float32)
        features = features_batch.to(device, dtype=torch.float32)
        labels = labels_batch.to(device, dtype=torch.long)
        
        # Forward pass
        predictions, survival_scores = model.forward(omics_data)
        
        # Calculate risk scores
        risk_scores = -torch.sum(survival_scores, dim=1)
        
        # Save statistics
        all_risk_scores.append(risk_scores.cpu().numpy().reshape(-1))
        all_censor_status.append(censor_status.cpu().numpy().reshape(-1))
        all_survival_time.append(survival_time.cpu().numpy().reshape(-1))
        
        # Update progress bar
        pbar.set_postfix({'batch': batch_idx + 1})
    
    # Calculate C-index
    try:
        c_index, _, _, _ = calculate_c_index(
            all_risk_scores, 
            all_censor_status, 
            all_survival_time
        )
        
        print(f"\nValidation Summary:")
        print(f"  C-index: {c_index:.4f}")
        print("-" * 40)
        
        return c_index
        
    except ValueError as e:
        print(f"\nWarning in validation: {e}")
        return None


if __name__ == "__main__":
    print('Start Training')
    best_c_index = 0
    logger = MetricLogger(config)
    for epoch in range(1, config.TRAIN.EPOCHS + 1):
        start_time = time.time()
        train_one_epoch(model, train_loader, optimizer)
        if config.TRAIN.LR_SCHEDULER.NAME == 'StepLR':
            lr_scheduler.step()
        val_c_index = validate_one_epoch(model, val_loader)
        if val_c_index > best_c_index:
            best_c_index = val_c_index
            torch.save(model.state_dict(), f"{config.CHECKPOINTS_PATH}/{config.CHECKPOINTS_NAME}")
        epoch_time = time.time() - start_time
        print("epoch:{}, time:{:.2f}s, best_c_index:{:.4f}\n".format(epoch, epoch_time, best_c_index), flush=True)
      
    print('\nStart Testing')
    test_acc = validate_one_epoch(model, test_loader)

