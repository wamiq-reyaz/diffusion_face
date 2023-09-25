from importlib import import_module

def get_conditioner(cfg):
    C = getattr(import_module(f'training_diffusion.conditioners.{cfg.conditioner.name}'), 'Conditioner')
    return C(cfg)