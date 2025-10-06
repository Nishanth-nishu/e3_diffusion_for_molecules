"""
Main interface for loading PDBbind dataset in the training pipeline.
"""

from pdbbind.data.pdbbind_dataset import create_pdbbind_dataloaders
from pathlib import Path


def retrieve_dataloaders(cfg):
    """
    Retrieve PDBbind dataloaders compatible with training pipeline.
    
    Args:
        cfg: Configuration object with attributes:
            - dataset: 'pdbbind'
            - batch_size: int
            - num_workers: int
            - data_dir: path to processed data
            - distance_cutoff: float
    
    Returns:
        dataloaders: dict with 'train', 'valid', 'test' keys
        charge_scale: None (not used for PDBbind)
    """
    data_dir = Path(cfg.data_dir) / 'processed'
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Processed data directory not found: {data_dir}\n"
            f"Please run preprocessing first:\n"
            f"  python pdbbind/data/preprocess_pdbbind.py "
            f"--pdbbind_dir <path> --output_dir {data_dir}"
        )
    
    dataloaders = create_pdbbind_dataloaders(
        data_dir=data_dir,
        batch_size=cfg.batch_size,
        distance_cutoff=cfg.distance_cutoff,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    # PDBbind doesn't use charge scaling like QM9
    charge_scale = None
    
    return dataloaders, charge_scale