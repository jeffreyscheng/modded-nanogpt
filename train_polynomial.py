import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import glob
import os
from collections import defaultdict
import numpy as np
import wandb
from typing import Dict, Any

class GeneralizedNewtonSchulz(nn.Module):
    def __init__(self, degree=3, num_iterations=5):
        """
        Initialize a generalized Newton-Schulz iteration module.
        
        Args:
            degree: Maximum degree of the matrix polynomial (n)
            num_iterations: Number of Newton-Schulz iterations to perform
        """
        super().__init__()
        self.degree = degree
        self.num_iterations = num_iterations
        
        # Create learnable coefficients for each term X(X^TX)^k
        # coefficient[k] corresponds to the term X(X^TX)^k
        # initialize with standard Gaussian
        self.coefficients = nn.Parameter(torch.randn(self.num_polynomial_terms))
        
        # ensure that the first coefficient is positive
        # because we know that we want to push small singular values the fastest
        with torch.no_grad():
            self.coefficients.data[0] = torch.abs(self.coefficients.data[0])
        
        # Learnable initial scaling factor
        self.initial_scale = nn.Parameter(torch.tensor(1.0))
    
    @property
    def num_polynomial_terms(self):
        """
        We only have odd terms in the polynomial, so the number of terms is (degree + 1) // 2
        """
        return (self.degree + 1) // 2

    def forward(self, X):
        """
        Forward pass implementing the generalized Newton-Schulz iteration
        
        Args:
            X: Input matrix of shape (batch_size, n, n)
        Returns:
            Result of generalized Newton-Schulz iteration
        """
        # Initial scaling
        frob_norm = torch.norm(X, p='fro', dim=(-2, -1), keepdim=True)
        X = X / frob_norm * torch.abs(self.initial_scale)

        assert not torch.isnan(X).any()
        
        # Newton-Schulz iterations
        for _ in range(self.num_iterations):
            terms = []
            XT = X.transpose(-2, -1)

            assert not torch.isnan(XT).any()
            
            # Compute X(X^TX)^k for k = 0 to degree-1
            current_term = X  # Starts with X
            XTX = torch.bmm(XT, X)  # Precompute X^TX
            
            for k in range(self.num_polynomial_terms):
                terms.append(current_term)
                # Compute next term by multiplying current term by X^TX
                current_term = torch.bmm(current_term, XTX)
                assert not torch.isnan(current_term).any(), f"NaN in term {k}"
            
            # Combine terms using learned coefficients
            X = sum(coeff * term for coeff, term in zip(self.coefficients, terms))
            
        return X

    def get_parameters(self):
        """Returns current parameter values"""
        return {
            'initial_scale': self.initial_scale.item(),
            'coefficients': self.coefficients.detach().cpu().tolist()
        }
    
    def print_polynomial(self):
        """
        Prints the current polynomial in a readable format
        """
        terms = []
        for i, coeff in enumerate(self.coefficients.detach().cpu().tolist()):
            if abs(coeff) > 1e-6:  # Only print significant terms
                if i == 0:
                    terms.append(f"{coeff:.3f}X")
                else:
                    terms.append(f"{coeff:.3f}X(X^TX)^{i}")
        
        return " + ".join(terms)

    def evaluate_scalar(self, x):
        """
        Evaluate the polynomial on scalar inputs
        Args:
            x: Input tensor of shape (N,)
        Returns:
            Result of polynomial evaluation
        """
        result = 0
        for i, coeff in enumerate(self.coefficients):
            # For scalar x, X(X^TX)^k becomes x * (x^2)^k = x^(2k+1)
            power = 2 * i + 1
            result = result + coeff * torch.pow(x, power)
        return result

class MatrixParamsDataset(Dataset):
    def __init__(self, checkpoint_dir, pattern="matrix_params_step*.pkl"):
        """
        Args:
            checkpoint_dir: Directory containing the checkpoint files
            pattern: Glob pattern to match checkpoint files
        """
        self.checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, pattern)))
        if not self.checkpoint_files:
            raise ValueError(f"No checkpoint files found in {checkpoint_dir} matching {pattern}")
        if len(self.checkpoint_files) < 2:
            raise ValueError(f"Need at least 2 checkpoints, found {len(self.checkpoint_files)}")
            
        # Skip the first checkpoint (batch 0)
        self.checkpoint_files = self.checkpoint_files[1:]
        
        # Load first non-zero checkpoint to get matrix names and sizes
        with open(self.checkpoint_files[0], 'rb') as f:
            first_checkpoint = pickle.load(f)
        
        # Create flat list of (name, shape, file_path) for indexing
        self.samples = [
            (name, shape, file_path)
            for file_path in self.checkpoint_files
            for name, _, shape in first_checkpoint
        ]

    def __getitem__(self, idx):
        name, shape, file_path = self.samples[idx]
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
            # Find the matrix with matching name
            for param_name, matrix, _ in params:
                if param_name == name:
                    return torch.from_numpy(matrix)
        raise ValueError(f"Matrix {name} not found in {file_path}")

    def __len__(self):
        return len(self.samples)


def collate_matrices(batch):
    """
    Custom collate function that only batches matrices of the same size
    """
    if not batch:
        return torch.tensor([])
    
    # All matrices in the batch should have the same size due to the batch_sampler
    return torch.stack(batch)

class SameSizeBatchSampler:
    """
    Custom BatchSampler that only batches matrices of the same size together,
    regardless of their location in the architecture
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Group indices by matrix shape only
        self.shape_to_indices = defaultdict(list)
        for idx, (_, shape, _) in enumerate(dataset.samples):
            self.shape_to_indices[shape].append(idx)
        
        # Convert lists to arrays for faster shuffling
        self.shape_to_indices = {
            shape: np.array(indices)
            for shape, indices in self.shape_to_indices.items()
        }

    def __iter__(self):
        # Shuffle indices within each shape group
        for indices in self.shape_to_indices.values():
            np.random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size].tolist()

    def __len__(self):
        return sum(len(indices) // self.batch_size + (1 if len(indices) % self.batch_size else 0)
                  for indices in self.shape_to_indices.values())

# Example usage:
def get_matrix_dataloader(checkpoint_dir, batch_size=8):
    dataset = MatrixParamsDataset(checkpoint_dir)
    batch_sampler = SameSizeBatchSampler(dataset, batch_size)
    
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_matrices,
        num_workers=4  # Adjust based on your system
    )

def loss_fn(matrix_batch):
    """
    Given a batch of matrices of shape (batch_size, n, m),
    penalize each singular value's deviation from 1
    by computing ||X^TX - I||^2
    """
    XT = matrix_batch.transpose(-2, -1)
    I = torch.eye(matrix_batch.size(-1)).to(matrix_batch.device)
    return torch.norm(XT @ matrix_batch - I, p='fro')

def create_fn_viz(fn, title, device):
    """
    plot f(x) = ax + bx^3 + cx^5 + ...
    use ns_poly.print_polynomial() as the plot title
    plot from x=-1 to x=3
    return an object that can be logged to wandb
    """
    fig, ax = plt.subplots()
    x = torch.linspace(-0.5, 2.0, 1000).to(device)
    y = fn(x)
    ax.plot(x.cpu().detach().numpy(), y.cpu().detach().numpy())
    ax.title(title)
    ax.ylim(-1, 3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)  # x-axis
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)  # y-axis
    img = wandb.Image(plt)
    plt.close()
    return img

def get_iterated_fn(ns_poly):
    def iterated_fn(x):
        y = x
        for _ in range(ns_poly.num_iterations):
            y = ns_poly.evaluate_scalar(y)
        return y
    return iterated_fn


def train_newton_schulz(config: Dict[Any, Any]):
    """
    Training function that accepts a config dictionary for hyperparameter sweeps
    """
    # Initialize wandb
    run = wandb.init(project="newton-schulz-polynomial", config=config)
    
    # Create model and optimizer with config parameters
    ns_poly = GeneralizedNewtonSchulz(
        degree=config['degree'],
        num_iterations=config['num_iterations']
    ).to(config['device'])
    
    meta_optimizer = torch.optim.Adam(
        ns_poly.parameters(),
        lr=config['learning_rate'],
        betas=config['adam_betas']
    )
    
    dataloader = get_matrix_dataloader(
        config['checkpoint_dir'],
        batch_size=config['batch_size']
    )
    
    for epoch in range(config['num_epochs']):
        epoch_loss = 0.0
        num_batches = 0
        
        for i, matrices in enumerate(dataloader):
            matrices = matrices.to(config['device'])
            zeropower_matrices = ns_poly(matrices)
            loss = loss_fn(zeropower_matrices)
            
            # Normalize loss by batch size for consistent scaling
            loss = loss / matrices.size(0)
            loss.backward()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if (i + 1) % config['accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(
                    ns_poly.parameters(),
                    max_norm=config['max_grad_norm']
                )
                meta_optimizer.step()
                meta_optimizer.zero_grad()
                
                # Log metrics
                wandb.log({
                    "batch_loss": loss.item(),
                    "polynomial_plot": create_fn_viz(ns_poly, ns_poly.print_polynomial(), config['device']),
                    "iterated_polynomial_plot": create_fn_viz(get_iterated_fn(ns_poly), ns_poly.print_polynomial(), config['device']),
                    "initial_scale": ns_poly.initial_scale.item()
                })
        
        # Log epoch metrics
        avg_epoch_loss = epoch_loss / num_batches
        wandb.log({
            "epoch": epoch,
            "avg_epoch_loss": avg_epoch_loss,
        })
    
    # Close wandb run
    run.finish()

if __name__ == "__main__":
    # Define default configuration
    default_config = {
        "device": "cuda",
        "checkpoint_dir": "logs/fa116850-b156-40fe-88b7-66ff6399d9be",
        "degree": 5,
        "num_iterations": 5,
        "learning_rate": 0.1,
        "adam_betas": (0.9, 0.9),
        "batch_size": 32,
        "accumulation_steps": 10,
        "max_grad_norm": 1.0,
        "num_epochs": 20
    }
    
    # For single run
    train_newton_schulz(default_config)
    
    # For hyperparameter sweep
    sweep_config = {
        'method': 'grid',  # or 'random', 'bayes'
        'parameters': {
            'degree': {'values': [3, 5, 7]},
            'num_iterations': {'values': [2, 3, 4, 5, 10]},
            'learning_rate': {'values': [0.001, 0.01, 0.1]},
        }
    }
    
    # Uncomment to run sweep
    sweep_id = wandb.sweep(sweep_config, project="newton-schulz-polynomial")
    wandb.agent(sweep_id, function=train_newton_schulz)