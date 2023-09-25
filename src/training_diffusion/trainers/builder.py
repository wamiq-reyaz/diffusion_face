from importlib import import_module


def get_trainer(cfg, model_builder, rank=0):
    Trainer = getattr(import_module(f'training_diffusion.trainers.{cfg.training.name}'), 'Trainer')
    return Trainer(cfg, model_builder, rank=rank)
