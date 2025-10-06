"""
PyTorch Dataset and DataLoader for PDBbind.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np


class PDBBindDataset(Dataset):
    """PyTorch Dataset for PDBbind protein-ligand complexes."""
    
    def __init__(self, data_file, transform=None):
        """
        Args:
            data_file: Path to preprocessed pickle file
            transform: Optional transform to apply
        """
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx].copy()
        
        if self.transform:
            item = self.transform(item)
        
        # Convert to tensors
        item['positions'] = torch.from_numpy(item['positions'])
        item['charges'] = torch.from_numpy(item['charges'])
        item['one_hot'] = torch.from_numpy(item['one_hot'])
        item['is_ligand'] = torch.from_numpy(item['is_ligand'])
        
        return item


class PDBBindCollate:
    """Collate function for batching variable-sized protein-ligand complexes."""
    
    def __init__(self, distance_cutoff=10.0):
        """
        Args:
            distance_cutoff: Distance threshold for edge creation (Angstroms)
        """
        self.distance_cutoff = distance_cutoff
    
    def __call__(self, batch):
        """
        Collate batch of protein-ligand complexes.
        
        Args:
            batch: List of dictionaries with keys:
                - positions: (N, 3)
                - charges: (N,)
                - one_hot: (N, n_atom_types)
                - is_ligand: (N,)
                - num_atoms: int
        
        Returns:
            Batched dictionary with padded tensors and masks
        """
        batch_size = len(batch)
        
        # Find max number of atoms in batch
        max_atoms = max([item['num_atoms'] for item in batch])
        n_atom_types = batch[0]['one_hot'].shape[1]
        
        # Initialize padded tensors
        positions = torch.zeros(batch_size, max_atoms, 3)
        charges = torch.zeros(batch_size, max_atoms, 1)
        one_hot = torch.zeros(batch_size, max_atoms, n_atom_types)
        is_ligand = torch.zeros(batch_size, max_atoms, 1)
        atom_mask = torch.zeros(batch_size, max_atoms, 1)
        
        # Fill in data
        for i, item in enumerate(batch):
            n = item['num_atoms']
            positions[i, :n] = item['positions']
            charges[i, :n, 0] = item['charges']
            one_hot[i, :n] = item['one_hot']
            is_ligand[i, :n, 0] = item['is_ligand']
            atom_mask[i, :n, 0] = 1
        
        # Create edge mask based on atom mask and distance
        edge_mask = self._create_edge_mask(positions, atom_mask, self.distance_cutoff)
        
        return {
            'positions': positions,
            'charges': charges,
            'one_hot': one_hot,
            'is_ligand': is_ligand,
            'atom_mask': atom_mask,
            'edge_mask': edge_mask,
            'num_atoms': torch.tensor([item['num_atoms'] for item in batch])
        }
    
    def _create_edge_mask(self, positions, atom_mask, cutoff):
        """
        Create edge mask based on distance cutoff.
        
        Args:
            positions: (batch_size, n_nodes, 3)
            atom_mask: (batch_size, n_nodes, 1)
            cutoff: Distance threshold
        
        Returns:
            edge_mask: (batch_size * n_nodes * n_nodes, 1)
        """
        batch_size, n_nodes, _ = positions.shape
        
        # Compute pairwise distances
        # Shape: (batch_size, n_nodes, n_nodes)
        pos_i = positions.unsqueeze(2)  # (bs, n, 1, 3)
        pos_j = positions.unsqueeze(1)  # (bs, 1, n, 3)
        distances = torch.norm(pos_i - pos_j, dim=-1)
        
        # Create base edge mask from atom mask
        # Shape: (batch_size, n_nodes, n_nodes)
        mask_i = atom_mask.squeeze(-1).unsqueeze(2)  # (bs, n, 1)
        mask_j = atom_mask.squeeze(-1).unsqueeze(1)  # (bs, 1, n)
        base_mask = mask_i * mask_j
        
        # Apply distance cutoff
        distance_mask = (distances <= cutoff).float()
        
        # Combine masks and remove self-loops
        diag_mask = 1 - torch.eye(n_nodes).unsqueeze(0)
        edge_mask = base_mask * distance_mask * diag_mask
        
        # Reshape to (batch_size * n_nodes * n_nodes, 1)
        edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
        
        return edge_mask


def create_pdbbind_dataloaders(data_dir, 
                               batch_size=8,
                               distance_cutoff=10.0,
                               num_workers=4,
                               pin_memory=True):
    """
    Create train/val/test dataloaders for PDBbind.
    
    Args:
        data_dir: Directory containing train.pkl, valid.pkl, test.pkl
        batch_size: Batch size
        distance_cutoff: Distance for edge creation
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory for GPU transfer
    
    Returns:
        Dictionary with 'train', 'valid', 'test' dataloaders
    """
    from pathlib import Path
    
    data_dir = Path(data_dir)
    collate_fn = PDBBindCollate(distance_cutoff=distance_cutoff)
    
    dataloaders = {}
    
    for split in ['train', 'valid', 'test']:
        data_file = data_dir / f'{split}.pkl'
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        dataset = PDBBindDataset(data_file)
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory
        )
    
    return dataloaders


class PDBBindTransform:
    """Optional transforms for PDBbind data augmentation."""
    
    def __init__(self, remove_mean=True, add_noise=0.0):
        """
        Args:
            remove_mean: Whether to center coordinates
            add_noise: Standard deviation of Gaussian noise to add
        """
        self.remove_mean = remove_mean
        self.add_noise = add_noise
    
    def __call__(self, item):
        """Apply transform to a single data item."""
        positions = item['positions']
        
        # Remove mean (center of mass)
        if self.remove_mean:
            positions = positions - positions.mean(axis=0, keepdims=True)
        
        # Add noise for data augmentation
        if self.add_noise > 0:
            noise = np.random.randn(*positions.shape) * self.add_noise
            positions = positions + noise
        
        item['positions'] = positions
        return item


def compute_dataset_statistics(data_dir):
    """
    Compute statistics over the dataset for normalization.
    
    Args:
        data_dir: Directory with processed data
    
    Returns:
        Dictionary with dataset statistics
    """
    from pathlib import Path
    
    data_dir = Path(data_dir)
    train_file = data_dir / 'train.pkl'
    
    with open(train_file, 'rb') as f:
        data = pickle.load(f)
    
    # Compute atom type distribution
    atom_type_counts = {}
    n_nodes_dist = {}
    all_distances = []
    
    for item in data:
        charges = item['charges']
        positions = item['positions']
        
        # Count atom types
        for charge in charges:
            atom_type_counts[int(charge)] = atom_type_counts.get(int(charge), 0) + 1
        
        # Count number of nodes
        n_nodes = len(charges)
        # Round to nearest 50 for binning
        n_nodes_bin = (n_nodes // 50) * 50
        n_nodes_dist[n_nodes_bin] = n_nodes_dist.get(n_nodes_bin, 0) + 1
        
        # Compute pairwise distances
        pos_i = positions[:, None, :]
        pos_j = positions[None, :, :]
        dists = np.linalg.norm(pos_i - pos_j, axis=-1)
        
        # Take upper triangle (avoid duplicates and self-loops)
        mask = np.triu(np.ones_like(dists), k=1).astype(bool)
        all_distances.extend(dists[mask].flatten())
    
    # Create distance histogram (100 bins)
    distance_hist, _ = np.histogram(all_distances, bins=100, range=(0, 15))
    
    stats = {
        'atom_types': atom_type_counts,
        'n_nodes': n_nodes_dist,
        'distances': distance_hist.tolist(),
        'n_samples': len(data)
    }
    
    return stats


if __name__ == '__main__':
    # Test the dataset and dataloader
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = 'data/pdbbind/processed'
    
    print(f"Loading data from {data_dir}")
    
    # Compute statistics
    print("\nComputing dataset statistics...")
    stats = compute_dataset_statistics(data_dir)
    print(f"Number of samples: {stats['n_samples']}")
    print(f"Atom type distribution: {stats['atom_types']}")
    print(f"Node count distribution: {stats['n_nodes']}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = create_pdbbind_dataloaders(
        data_dir,
        batch_size=4,
        distance_cutoff=10.0,
        num_workers=0
    )
    
    # Test loading a batch
    print("\nTesting dataloader...")
    for split in ['train', 'valid', 'test']:
        batch = next(iter(dataloaders[split]))
        print(f"\n{split.upper()} batch:")
        print(f"  Positions shape: {batch['positions'].shape}")
        print(f"  Charges shape: {batch['charges'].shape}")
        print(f"  One-hot shape: {batch['one_hot'].shape}")
        print(f"  Is ligand shape: {batch['is_ligand'].shape}")
        print(f"  Atom mask shape: {batch['atom_mask'].shape}")
        print(f"  Edge mask shape: {batch['edge_mask'].shape}")
        print(f"  Num atoms: {batch['num_atoms']}")
        
        # Check edge mask statistics
        edge_mask = batch['edge_mask']
        n_edges = edge_mask.sum().item()
        total_possible = edge_mask.numel()
        print(f"  Active edges: {n_edges}/{total_possible} ({100*n_edges/total_possible:.2f}%)")