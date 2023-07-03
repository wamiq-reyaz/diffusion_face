import os 
import socket
from typing import Dict

def get_config() -> Dict:
    # whether locally or on IBEX
    hostname = socket.getfqdn()
    if 'ibex' in hostname:
        PREFIX = '/ibex/ai/home/parawr/Projects/diffusion/'
    else:
        PREFIX = '/mnt/ibex_ai/Projects/diffusion/'

    # everything has a common suffix and directory structure
    CONFIG = {
        'DATA_DIR':         os.path.join(PREFIX, 'data'), 
        'SCRATCH_DIR':      os.path.join(PREFIX, 'scratch'), 
        'RESULTS_DIR':      os.path.join(PREFIX, 'results'), 
        'WANDB_PROJECT':    'morph_diffusion',
    }

    return CONFIG

__all__ = [get_config]
