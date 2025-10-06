"""
Main training script for PDBbind dataset with EDM.
"""

import copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from pdbbind import dataset
from qm9.models import get_optim, get_model
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from qm9.utils import prepare_context, compute_mean_mad
from train_test import train_epoch, test, analyze_and_save
import yaml

parser = argparse.ArgumentParser(description='EDM for PDBbind')
parser.add_argument('--config', type=str, default='configs/pdbbind_config.yaml',
                    help='Path to config file')
parser.add_argument('--exp_name', type=str, default=None,
                    help='Experiment name (overrides config)')
parser.add_argument('--resume', type=str, default=None,
                    help='Path to checkpoint to resume from')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disable CUDA')
parser.add_argument('--no_wandb', action='store_true',
                    help='Disable wandb logging')

args = parser.parse_args()

# Load configuration from YAML
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Convert config dict to argparse Namespace for compatibility
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

cfg = Config(**config)

# Override with command line arguments
if args.exp_name:
    cfg.exp_name = args.exp_name
if args.resume:
    cfg.resume = args.resume
if args.no_cuda:
    cfg.no_cuda = True
if args.no_wandb:
    cfg.no_wandb = True

# Add dataset info to config
dataset_info = get_dataset_info('pdbbind', cfg.get('remove_h', False))
cfg.dataset_info = dataset_info

atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

cfg.wandb_usr = utils.get_wandb_username(getattr(cfg, 'wandb_usr', None))

cfg.cuda = not cfg.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cfg.cuda else "cpu")
dtype = torch.float32

# Handle resume
if cfg.resume is not None:
    exp_name = cfg.exp_name + '_resume'
    start_epoch = cfg.start_epoch
    resume = cfg.resume
    wandb_usr = cfg.wandb_usr
    
    with open(join(cfg.resume, 'args.pickle'), 'rb') as f:
        cfg = pickle.load(f)
    
    cfg.resume = resume
    cfg.break_train_epoch = False
    cfg.exp_name = exp_name
    cfg.start_epoch = start_epoch
    cfg.wandb_usr = wandb_usr
    
    print("Resumed config:")
    print(cfg)

utils.create_folders(cfg)

# Wandb setup
if cfg.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if cfg.online else 'offline'

kwargs = {
    'entity': cfg.wandb_usr,
    'name': cfg.exp_name,
    'project': 'pdbbind_edm',
    'config': vars(cfg),
    'settings': wandb.Settings(_disable_stats=True),
    'reinit': True,
    'mode': mode
}
wandb.init(**kwargs)
wandb.save('*.txt')

# Retrieve PDBbind dataloaders
print("Loading PDBbind dataset...")
dataloaders, charge_scale = dataset.retrieve_dataloaders(cfg)

data_dummy = next(iter(dataloaders['train']))

# Handle conditioning (e.g., on binding affinity)
if len(cfg.conditioning) > 0:
    print(f'Conditioning on {cfg.conditioning}')
    property_norms = compute_mean_mad(dataloaders, cfg.conditioning, cfg.dataset)
    context_dummy = prepare_context(cfg.conditioning, data_dummy, property_norms)
    context_node_nf = context_dummy.size(2)
else:
    context_node_nf = 0
    property_norms = None

cfg.context_node_nf = context_node_nf

# Add is_ligand feature to node features
# The model will receive one_hot + charges + is_ligand as input
print(f"Node feature dimensions:")
print(f"  One-hot: {len(dataset_info['atom_decoder'])}")
print(f"  Charges: {1 if cfg.include_charges else 0}")
print(f"  Is ligand: 1")
print(f"  Total: {len(dataset_info['atom_decoder']) + (1 if cfg.include_charges else 0) + 1}")

# Create model
model, nodes_dist, prop_dist = get_model(cfg, device, dataset_info, dataloaders['train'])
if prop_dist is not None:
    prop_dist.set_normalizer(property_norms)
model = model.to(device)
optim = get_optim(cfg, model)

gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)

def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)

def main():
    if cfg.resume is not None:
        flow_state_dict = torch.load(join(cfg.resume, 'flow.npy'))
        optim_state_dict = torch.load(join(cfg.resume, 'optim.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)
    
    # Initialize dataparallel if enabled and possible
    if cfg.dp and torch.cuda.device_count() > 1:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model
    
    # Initialize EMA
    if cfg.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(cfg.ema_decay)
        
        if cfg.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp
    
    best_nll_val = 1e8
    best_nll_test = 1e8
    
    for epoch in range(cfg.start_epoch, cfg.n_epochs):
        start_epoch = time.time()
        
        train_epoch(
            args=cfg,
            loader=dataloaders['train'],
            epoch=epoch,
            model=model,
            model_dp=model_dp,
            model_ema=model_ema,
            ema=ema,
            device=device,
            dtype=dtype,
            property_norms=property_norms,
            nodes_dist=nodes_dist,
            dataset_info=dataset_info,
            gradnorm_queue=gradnorm_queue,
            optim=optim,
            prop_dist=prop_dist
        )
        
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")
        
        if epoch % cfg.test_epochs == 0:
            if isinstance(model, en_diffusion.EnVariationalDiffusion):
                wandb.log(model.log_info(), commit=True)
            
            if not cfg.break_train_epoch:
                analyze_and_save(
                    args=cfg,
                    epoch=epoch,
                    model_sample=model_ema,
                    nodes_dist=nodes_dist,
                    dataset_info=dataset_info,
                    device=device,
                    prop_dist=prop_dist,
                    n_samples=cfg.n_stability_samples
                )
            
            nll_val = test(
                args=cfg,
                loader=dataloaders['valid'],
                epoch=epoch,
                eval_model=model_ema_dp,
                partition='Val',
                device=device,
                dtype=dtype,
                nodes_dist=nodes_dist,
                property_norms=property_norms
            )
            
            nll_test = test(
                args=cfg,
                loader=dataloaders['test'],
                epoch=epoch,
                eval_model=model_ema_dp,
                partition='Test',
                device=device,
                dtype=dtype,
                nodes_dist=nodes_dist,
                property_norms=property_norms
            )
            
            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test
                
                if cfg.save_model:
                    cfg.current_epoch = epoch + 1
                    utils.save_model(optim, 'outputs/%s/optim.npy' % cfg.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model.npy' % cfg.exp_name)
                    if cfg.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % cfg.exp_name)
                    with open('outputs/%s/args.pickle' % cfg.exp_name, 'wb') as f:
                        pickle.dump(cfg, f)
            
            if cfg.save_model and epoch % 10 == 0:
                utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (cfg.exp_name, epoch))
                utils.save_model(model, 'outputs/%s/generative_model_%d.npy' % (cfg.exp_name, epoch))
                if cfg.ema_decay > 0:
                    utils.save_model(model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (cfg.exp_name, epoch))
                with open('outputs/%s/args_%d.pickle' % (cfg.exp_name, epoch), 'wb') as f:
                    pickle.dump(cfg, f)
            
            print('Val loss: %.4f \t Test loss: %.4f' % (nll_val, nll_test))
            print('Best val loss: %.4f \t Best test loss: %.4f' % (best_nll_val, best_nll_test))
            wandb.log({"Val loss": nll_val}, commit=True)
            wandb.log({"Test loss": nll_test}, commit=True)
            wandb.log({"Best cross-validated test loss": best_nll_test}, commit=True)


if __name__ == "__main__":
    main()