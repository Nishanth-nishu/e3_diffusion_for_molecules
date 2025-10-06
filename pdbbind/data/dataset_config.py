# PDBbind dataset configuration
# This should be added to configs/datasets_config.py

pdbbind_config = {
    'name': 'pdbbind',
    # Combined protein + ligand atoms
    'atom_encoder': {
        'C': 0, 'N': 1, 'O': 2, 'S': 3,      # Common in both
        'F': 4, 'Cl': 5, 'Br': 6, 'P': 7,    # Ligand-specific
        'H': 8                                 # Hydrogen
    },
    'atom_decoder': ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P', 'H'],
    
    # Reasonable distribution for protein-ligand complexes
    # These will be computed from actual data during preprocessing
    'n_nodes': {
        50: 100, 100: 200, 150: 300, 200: 400, 250: 500,
        300: 600, 350: 500, 400: 400, 450: 300, 500: 200,
        550: 150, 600: 100, 650: 80, 700: 60, 750: 40,
        800: 30, 850: 20, 900: 15, 950: 10, 1000: 5
    },
    
    'max_n_nodes': 1000,  # Maximum atoms in complex
    
    # Placeholder atom type distribution (will be computed)
    'atom_types': {
        0: 500000,  # C
        1: 150000,  # N
        2: 200000,  # O
        3: 20000,   # S
        4: 5000,    # F
        5: 3000,    # Cl
        6: 1000,    # Br
        7: 2000,    # P
        8: 300000   # H
    },
    
    # Distance distribution (placeholder)
    'distances': [1000] * 100,
    
    # Visualization colors for each atom type
    'colors_dic': ['C7', 'C0', 'C3', 'C8', 'C1', 'C4', 'C6', 'C5', '#FFFFFF99'],
    'radius_dic': [0.77, 0.77, 0.77, 1.0, 0.77, 1.0, 1.0, 1.0, 0.46],
    
    'with_h': True,
    
    # PDBbind-specific
    'is_protein_ligand': True,
    'distance_cutoff': 10.0,  # Angstroms
    'pocket_cutoff': 10.0,     # Distance to define binding pocket
}

pdbbind_no_h = {
    'name': 'pdbbind',
    'atom_encoder': {
        'C': 0, 'N': 1, 'O': 2, 'S': 3,
        'F': 4, 'Cl': 5, 'Br': 6, 'P': 7
    },
    'atom_decoder': ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P'],
    'max_n_nodes': 1000,
    'n_nodes': {
        50: 100, 100: 200, 150: 300, 200: 400, 250: 500,
        300: 600, 350: 500, 400: 400, 450: 300, 500: 200
    },
    'atom_types': {
        0: 500000, 1: 150000, 2: 200000, 3: 20000,
        4: 5000, 5: 3000, 6: 1000, 7: 2000
    },
    'distances': [1000] * 100,
    'colors_dic': ['C7', 'C0', 'C3', 'C8', 'C1', 'C4', 'C6', 'C5'],
    'radius_dic': [0.77, 0.77, 0.77, 1.0, 0.77, 1.0, 1.0, 1.0],
    'with_h': False,
    'is_protein_ligand': True,
    'distance_cutoff': 10.0,
    'pocket_cutoff': 10.0,
}


def get_pdbbind_config(remove_h=False):
    """Get PDBbind dataset configuration."""
    if remove_h:
        return pdbbind_no_h
    return pdbbind_config