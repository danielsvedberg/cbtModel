import pickle as pkl
import plotting_functions as pf
from model_functions import *
from config_script import *

# Load existing parameters instead of generating new ones
print("Loading existing parameters from params_nm.pkl...")
try:
    with open('params_nm.pkl', 'rb') as f:
        params_nm = pkl.load(f)
except FileNotFoundError:
    print("Error: params_nm.pkl not found! Please run training_script.py first.")
    exit(1)

# Generate inputs, outputs, and masks for the task
all_inputs, all_outputs, all_masks = self_timed_movement_task(
    config['T_start'], config['T_cue'], config['T_wait'], config['T_movement'], config['T']
)

#### CONTINUE TRAINING ###############

# Train using the loaded parameters as the starting point instead of `params`
params_nm, losses_nm = fit_nm_rnn(
    all_inputs, all_outputs, all_masks,
    params_nm,  # <--- CRITICAL: using loaded params here
    optimizer, 
    x0, z0, 
    config['num_full_train_iters'],
    config['tau_x'], config['tau_z'], 
    wandb_log=False, 
    modulation=True
)

# Save params_nm back to the pickle file
with open('params_nm.pkl', 'wb') as f:
    pkl.dump(params_nm, f)

pf.plot_loss(losses_nm)
print(f"Training complete. Continued for {config['num_full_train_iters']} iterations.")
