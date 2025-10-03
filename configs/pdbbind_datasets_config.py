# Protein amino acids (20 standard + special tokens)
PROTEIN_ATOMS = ['C', 'N', 'O', 'S']  # Simplified protein atoms
LIGAND_ATOMS = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']  # Common ligand atoms

pdbbind_config = {
    'name': 'pdbbind',
    # Combined atom encoder: protein atoms + ligand atoms + special token for protein/ligand
    'atom_encoder': {
        'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 
        'P': 5, 'Cl': 6, 'Br': 7, 'I': 8
    },
    'atom_decoder': ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'],
    
    # Node type feature: 0 = protein, 1 = ligand
    'node_type_encoder': {'protein': 0, 'ligand': 1},
    
    'max_n_nodes': 1100,  # max_protein_atoms + max_ligand_atoms
    
    # These will be computed from the dataset
    'n_nodes': {},
    'atom_types': {},
    'distances': [],
    
    'colors_dic': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
    'radius_dic': [0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77],
    'with_h': True,
    
    # Distance cutoff for edge creation
    'distance_cutoff': 10.0,
}

def get_dataset_info(dataset_name, remove_h=False):
    if dataset_name == 'pdbbind':
        config = pdbbind_config.copy()
        config['remove_h'] = remove_h
        return config
    else:
        # Fall back to original implementation
        from configs.datasets_config import get_dataset_info as original_get_dataset_info
        return original_get_dataset_info(dataset_name, remove_h)
