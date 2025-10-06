"""
Preprocess PDBbind data for training EDM.

This script:
1. Reads protein PDB files and ligand SDF/MOL2 files
2. Extracts binding pocket atoms based on distance cutoff
3. Creates unified graph representation
4. Saves preprocessed data in PyTorch format
"""

import os
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser, Selection
import logging
from tqdm import tqdm
import pickle
from pathlib import Path


class PDBBindPreprocessor:
    """Preprocess PDBbind dataset for EDM training."""
    
    def __init__(self, 
                 pdbbind_dir,
                 output_dir,
                 pocket_cutoff=10.0,
                 distance_cutoff=10.0,
                 remove_h=False,
                 max_protein_atoms=1000,
                 max_ligand_atoms=100,
                 pocket_only=True):
        """
        Args:
            pdbbind_dir: Path to PDBbind dataset
            output_dir: Where to save processed data
            pocket_cutoff: Distance from ligand to define pocket (Angstroms)
            distance_cutoff: Distance for edge creation (Angstroms)
            remove_h: Whether to remove hydrogen atoms
            max_protein_atoms: Maximum protein atoms to include
            max_ligand_atoms: Maximum ligand atoms
            pocket_only: If True, only use pocket atoms; else use full protein
        """
        self.pdbbind_dir = Path(pdbbind_dir)
        self.output_dir = Path(output_dir)
        self.pocket_cutoff = pocket_cutoff
        self.distance_cutoff = distance_cutoff
        self.remove_h = remove_h
        self.max_protein_atoms = max_protein_atoms
        self.max_ligand_atoms = max_ligand_atoms
        self.pocket_only = pocket_only
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Atom types we care about
        if remove_h:
            self.valid_atoms = {'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P'}
        else:
            self.valid_atoms = {'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P', 'H'}
        
        self.atom_encoder = {atom: i for i, atom in enumerate(sorted(self.valid_atoms))}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def parse_protein(self, pdb_file):
        """Parse protein structure and extract atom information."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        
        atom_positions = []
        atom_types = []
        
        for atom in Selection.unfold_entities(structure, 'A'):
            element = atom.element.strip().upper()
            
            # Skip if not in valid atoms
            if element not in self.valid_atoms:
                continue
            
            # Get coordinates
            coord = atom.get_coord()
            atom_positions.append(coord)
            atom_types.append(element)
        
        return np.array(atom_positions), atom_types
    
    def parse_ligand(self, ligand_file):
        """Parse ligand structure from SDF or MOL2 file."""
        # Try reading as SDF first, then MOL2
        if ligand_file.suffix == '.sdf':
            mol = Chem.SDMolSupplier(str(ligand_file), removeHs=self.remove_h)[0]
        elif ligand_file.suffix == '.mol2':
            mol = Chem.MolFromMol2File(str(ligand_file), removeHs=self.remove_h)
        else:
            raise ValueError(f"Unsupported ligand format: {ligand_file.suffix}")
        
        if mol is None:
            return None, None
        
        # Get 3D coordinates
        conf = mol.GetConformer()
        atom_positions = []
        atom_types = []
        
        for atom in mol.GetAtoms():
            element = atom.GetSymbol()
            
            if element not in self.valid_atoms:
                continue
            
            pos = conf.GetAtomPosition(atom.GetIdx())
            atom_positions.append([pos.x, pos.y, pos.z])
            atom_types.append(element)
        
        return np.array(atom_positions), atom_types
    
    def extract_pocket(self, protein_pos, ligand_pos):
        """Extract pocket atoms within cutoff distance of ligand."""
        if not self.pocket_only:
            return np.arange(len(protein_pos))
        
        # Calculate distances from each protein atom to any ligand atom
        distances = np.linalg.norm(
            protein_pos[:, None, :] - ligand_pos[None, :, :], 
            axis=2
        )
        
        # Find protein atoms within cutoff
        min_distances = distances.min(axis=1)
        pocket_indices = np.where(min_distances <= self.pocket_cutoff)[0]
        
        return pocket_indices
    
    def process_complex(self, pdb_id, protein_file, ligand_file):
        """Process a single protein-ligand complex."""
        try:
            # Parse protein and ligand
            protein_pos, protein_types = self.parse_protein(protein_file)
            ligand_pos, ligand_types = self.parse_ligand(ligand_file)
            
            if ligand_pos is None or len(ligand_pos) == 0:
                self.logger.warning(f"Failed to parse ligand for {pdb_id}")
                return None
            
            # Extract pocket
            pocket_indices = self.extract_pocket(protein_pos, ligand_pos)
            
            if len(pocket_indices) == 0:
                self.logger.warning(f"No pocket atoms found for {pdb_id}")
                return None
            
            # Subsample if too many atoms
            if len(pocket_indices) > self.max_protein_atoms:
                pocket_indices = np.random.choice(
                    pocket_indices, 
                    self.max_protein_atoms, 
                    replace=False
                )
            
            pocket_pos = protein_pos[pocket_indices]
            pocket_types = [protein_types[i] for i in pocket_indices]
            
            # Subsample ligand if needed
            if len(ligand_pos) > self.max_ligand_atoms:
                ligand_indices = np.random.choice(
                    len(ligand_pos),
                    self.max_ligand_atoms,
                    replace=False
                )
                ligand_pos = ligand_pos[ligand_indices]
                ligand_types = [ligand_types[i] for i in ligand_indices]
            
            # Combine protein and ligand
            positions = np.vstack([pocket_pos, ligand_pos])
            atom_types = pocket_types + ligand_types
            
            # Create is_ligand mask (0 for protein, 1 for ligand)
            is_ligand = np.zeros(len(positions), dtype=np.int64)
            is_ligand[len(pocket_pos):] = 1
            
            # Encode atom types
            charges = np.array([self.atom_encoder[at] for at in atom_types])
            
            # Create one-hot encoding
            n_atom_types = len(self.atom_encoder)
            one_hot = np.zeros((len(charges), n_atom_types))
            one_hot[np.arange(len(charges)), charges] = 1
            
            # Store data
            data = {
                'positions': positions.astype(np.float32),
                'charges': charges.astype(np.int64),
                'one_hot': one_hot.astype(np.float32),
                'is_ligand': is_ligand,
                'num_atoms': len(positions),
                'pdb_id': pdb_id
            }
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error processing {pdb_id}: {str(e)}")
            return None
    
    def process_dataset(self, split='refined-set', indices_file=None):
        """
        Process entire PDBbind dataset.
        
        Args:
            split: 'refined-set' or 'general-set'
            indices_file: Optional file with PDB IDs to process
        """
        split_dir = self.pdbbind_dir / split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        # Get list of complexes to process
        if indices_file and os.path.exists(indices_file):
            with open(indices_file) as f:
                pdb_ids = [line.strip() for line in f]
        else:
            pdb_ids = [d.name for d in split_dir.iterdir() if d.is_dir()]
        
        self.logger.info(f"Processing {len(pdb_ids)} complexes from {split}")
        
        processed_data = []
        failed_ids = []
        
        for pdb_id in tqdm(pdb_ids, desc="Processing complexes"):
            complex_dir = split_dir / pdb_id
            
            # Find protein and ligand files
            protein_file = complex_dir / f"{pdb_id}_protein.pdb"
            
            # Try different ligand file formats
            ligand_file = None
            for ext in ['.sdf', '.mol2']:
                candidate = complex_dir / f"{pdb_id}_ligand{ext}"
                if candidate.exists():
                    ligand_file = candidate
                    break
            
            if not protein_file.exists() or ligand_file is None:
                self.logger.warning(f"Missing files for {pdb_id}")
                failed_ids.append(pdb_id)
                continue
            
            # Process complex
            data = self.process_complex(pdb_id, protein_file, ligand_file)
            
            if data is not None:
                processed_data.append(data)
            else:
                failed_ids.append(pdb_id)
        
        self.logger.info(f"Successfully processed: {len(processed_data)}/{len(pdb_ids)}")
        self.logger.info(f"Failed: {len(failed_ids)}")
        
        return processed_data, failed_ids
    
    def save_processed_data(self, data, split_name='train'):
        """Save processed data to disk."""
        output_file = self.output_dir / f"{split_name}.pkl"
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Saved {len(data)} complexes to {output_file}")
    
    def split_data(self, data, val_prop=0.1, test_prop=0.1, seed=42):
        """Split data into train/val/test sets."""
        np.random.seed(seed)
        n = len(data)
        
        indices = np.random.permutation(n)
        
        n_test = int(n * test_prop)
        n_val = int(n * val_prop)
        n_train = n - n_test - n_val
        
        train_data = [data[i] for i in indices[:n_train]]
        val_data = [data[i] for i in indices[n_train:n_train+n_val]]
        test_data = [data[i] for i in indices[n_train+n_val:]]
        
        return train_data, val_data, test_data


def main():
    """Main preprocessing script."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdbbind_dir', type=str, required=True,
                       help='Path to PDBbind dataset')
    parser.add_argument('--output_dir', type=str, default='data/pdbbind/processed',
                       help='Output directory')
    parser.add_argument('--split', type=str, default='refined-set',
                       choices=['refined-set', 'general-set'])
    parser.add_argument('--pocket_cutoff', type=float, default=10.0)
    parser.add_argument('--distance_cutoff', type=float, default=10.0)
    parser.add_argument('--remove_h', action='store_true')
    parser.add_argument('--max_protein_atoms', type=int, default=1000)
    parser.add_argument('--max_ligand_atoms', type=int, default=100)
    parser.add_argument('--pocket_only', action='store_true', default=True)
    parser.add_argument('--val_prop', type=float, default=0.1)
    parser.add_argument('--test_prop', type=float, default=0.1)
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = PDBBindPreprocessor(
        pdbbind_dir=args.pdbbind_dir,
        output_dir=args.output_dir,
        pocket_cutoff=args.pocket_cutoff,
        distance_cutoff=args.distance_cutoff,
        remove_h=args.remove_h,
        max_protein_atoms=args.max_protein_atoms,
        max_ligand_atoms=args.max_ligand_atoms,
        pocket_only=args.pocket_only
    )
    
    # Process dataset
    data, failed = preprocessor.process_dataset(split=args.split)
    
    # Split into train/val/test
    train_data, val_data, test_data = preprocessor.split_data(
        data, 
        val_prop=args.val_prop,
        test_prop=args.test_prop
    )
    
    # Save splits
    preprocessor.save_processed_data(train_data, 'train')
    preprocessor.save_processed_data(val_data, 'valid')
    preprocessor.save_processed_data(test_data, 'test')
    
    print(f"\nDataset split:")
    print(f"Train: {len(train_data)}")
    print(f"Valid: {len(val_data)}")
    print(f"Test: {len(test_data)}")


if __name__ == '__main__':
    main()