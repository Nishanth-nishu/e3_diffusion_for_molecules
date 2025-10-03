# Experiment settings
exp_name: pdbbind_baseline
dataset: 'pdbbind'
filter_n_atoms: null
n_report_steps: 10
wandb_usr: your_username
no_cuda: False
wandb: True
online: True
data_dir: 'data/pdbbind'

# PDBbind specific settings
pdbbind_data_path: 'data/pdbbind/raw'
pdbbind_index_file: 'data/pdbbind/index/INDEX_general_PL_data.2020'
distance_cutoff: 10.0  # Angstroms for edge creation
max_protein_atoms: 1000
max_ligand_atoms: 100
remove_h_protein: True  # Remove hydrogens from protein
remove_h_ligand: False  # Keep hydrogens in ligand

# Training settings
n_epochs: 200
batch_size: 8  # Smaller due to larger molecules
lr: 0.0001
brute_force: False
break_train_epoch: False
dp: True
condition_time: True
clip_grad: True
save_model: True
generate_epochs: 10
num_workers: 4
test_epochs: 5
data_augmentation: True  # Rotation augmentation
resume: null
start_epoch: 0
ema_decay: 0.999
augment_noise: 0.0

# Model settings
model: 'egnn_dynamics'
probabilistic_model: 'diffusion'
diffusion_steps: 500
diffusion_noise_schedule: 'polynomial_2'
diffusion_loss_type: 'l2'
n_layers: 8  # More layers for complex structures
nf: 128
ode_regularization: 0.001
trace: 'hutch'
dequantization: 'argmax_variational'
tanh: True
attention: True
x_aggregation: 'sum'
conditioning: []  # Can add: binding_affinity, etc.
actnorm: True
norm_constant: 1
normalization_factor: 1
aggregation_method: 'sum'

# Placeholders
context_node_nf: 0
include_charges: True
