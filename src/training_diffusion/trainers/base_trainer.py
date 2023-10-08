# File author: Wamiq Reyaz
# email: wamiq.para@kaust.edu.sa

import os
import glob
import uuid
import warnings
from datetime import datetime as dt
from typing import Dict
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from ema_pytorch import EMA

class colors:
    '''Colors class:
    Reset all colors with colors.reset
    Two subclasses fg for foreground and bg for background.
    Use as colors.subclass.colorname.
    i.e. colors.fg.red or colors.bg.green
    Also, the generic bold, disable, underline, reverse, strikethrough,
    and invisible work with the main class
    i.e. colors.bold
    '''
    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'

    class fg:
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'

    class bg:
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'


def printc(text, color):
    print(f"{color}{text}{colors.reset}")

# ------------------------------------------------------------------------------

class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg

# ------------------------------------------------------------------------------

class RunningAverageDict:
    """A dictionary of running averages."""
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}


def is_rank_zero(rank):
    return rank == 0


class BaseTrainer:
    def __init__(self,
                 cfg,
                 model_builder,
                 rank):
        """ Base Trainer class for training a model."""
        
        self.cfg = cfg
        self.rank = rank

        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        model_builder = model_builder
        self.model = model_builder.get_model()
        self.conditioner = model_builder.get_conditioner()
        self.train_loader = model_builder.get_train_loader()
        self.test_loader = model_builder.get_test_loader()
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()
        self.ema = None
        if self.rank == 0:
            if self.cfg.training.get('ema', False):
                self.ema = EMA(self.model.module if self.cfg.num_gpus > 1 else self.model,
                                beta=self.cfg.training.ema_beta,
                                update_every=self.cfg.training.ema_update_every,
                                update_after_step=self.cfg.training.ema_update_after_step
                            ).cuda()
    
    def resize_to_target(self, prediction, target):
        if prediction.shape[2:] != target.shape[-2:]:
            prediction = nn.functional.interpolate(
                prediction, size=target.shape[-2:], mode="bilinear", align_corners=True
            )
        return prediction

    def load_ckpt(self, checkpoint_dir="./checkpoints", ckpt_type="best"):
        import glob
        import os

        if hasattr(self.config, "checkpoint"):
            checkpoint = self.config.checkpoint
        elif hasattr(self.config, "ckpt_pattern"):
            pattern = self.config.ckpt_pattern
            matches = glob.glob(os.path.join(
                checkpoint_dir, f"*{pattern}*{ckpt_type}*"))
            if not (len(matches) > 0):
                raise ValueError(f"No matches found for the pattern {pattern}")
            checkpoint = matches[0]
        else:
            return
        model = load_wts(self.model, checkpoint)
        # TODO : Resuming training is not properly supported in this repo. Implement loading / saving of optimizer and scheduler to support it.
        print("Loaded weights from {0}".format(checkpoint))
        warnings.warn(
            "Resuming training is not properly supported in this repo. Implement loading / saving of optimizer and scheduler to support it.")
        self.model = model

    def init_optimizer(self):
        m = self.model.module if self.cfg.num_gpus > 1 else self.model

        # if self.config.same_lr:
        #     print("Using same LR")
        #     if hasattr(m, 'core'):
        #         m.core.unfreeze()
        #     params = self.model.parameters()
        # else:
        #     print("Using diff LR")
        #     if not hasattr(m, 'get_lr_params'):
        #         raise NotImplementedError(
        #             f"Model {m.__class__.__name__} does not implement get_lr_params. Please implement it or use the same LR for all parameters.")

        #     params = m.get_lr_params(self.config.lr)
        params = m.parameters()

        return optim.Adam(params, lr=self.cfg.training.lr)

    def init_scheduler(self):
        lrs = [l['lr'] for l in self.optimizer.param_groups]
        return optim.lr_scheduler.OneCycleLR(self.optimizer,
                                            max_lr=lrs,
                                            total_steps=self.cfg.training.total_iter,
                                            cycle_momentum=self.cfg.training.cycle_momentum,
                                            base_momentum=0.85,
                                            max_momentum=0.95,
                                            div_factor=self.cfg.training.div_factor,
                                            final_div_factor=self.cfg.training.final_div_factor,
                                            pct_start=self.cfg.training.pct_start,
                                            three_phase=self.cfg.training.three_phase)

    def train_on_batch(self, batch, train_step):
        self.optimizer.zero_grad()
        loss = self.model(batch['data'], condition=None)
        for k, v in loss.items():
            loss[k].backward()
        self.optimizer.step()
        return loss 
    
    def validate_on_batch(self, batch, val_step):
        raise NotImplementedError # TODO: implement returning a dict of metrics and losses

    def raise_if_nan(self, losses):
        for key, value in losses.items():
            if torch.isnan(value):
                raise ValueError(f"{key} is NaN, Stopping training")

    # @property
    # def iters_per_epoch(self):
    #     return len(self.train_loader)

    @property
    def total_iter(self):
        return self.cfg.training.total_iter

    def should_early_stop(self):
        if self.config.get('early_stop', False) and self.step > self.config.early_stop:
            return True

    def train(self):
        if is_rank_zero(self.rank):
            print('Training...')
        self.uid = str(uuid.uuid4()).split('-')[-1]
        run_id = f"{dt.now().strftime('%Y-%m-%d__%H:%M:%S')}-{self.uid}"
        self.run_id = run_id
        experiment_dir = os.path.basename(self.cfg.experiment_dir)
        self.experiment_id = f"{experiment_dir}_{run_id}"
        self.should_write = (self.rank == 0)
        self.should_log = self.should_write  # and logging
        # convert omegaconf to dict
        cfg_to_log = OmegaConf.to_container(self.cfg, resolve=True)
        if self.should_log:
            wandb.init(project=self.cfg.project, name=self.experiment_id, config=cfg_to_log, dir=self.cfg.experiment_dir,
                       notes=self.cfg.notes, settings=wandb.Settings(start_method="fork"))

        self.model.train()
        if self.conditioner:
            self.conditioner.train()
        self.step = 0
        best_loss = np.inf
        validate_every = int(self.cfg.training.val_freq)

        losses = {}
        def stringify_losses(L): return "; ".join(map(
            lambda kv: f"{colors.fg.purple}{kv[0]}{colors.reset}: {round(kv[1].item(),3):.4e}", L.items()))
        
        _iterator = tqdm(range(self.step, self.total_iter),
                        desc=f"Iter: 0/{self.total_iter}. Loop: Train") \
                        if is_rank_zero(self.rank) \
                        else range(self.step, self.total_iter)
        dloader = iter(self.train_loader)
        for _iter in _iterator:
            # ------------------------------------------------------------------------------------------------------
            # Training loop
            # ------------------------------------------------------------------------------------------------------

            batch = next(dloader)

            losses = self.train_on_batch(batch, _iter)

            self.raise_if_nan(losses)
            if is_rank_zero(self.rank):
                _iterator.set_description(
                    f"Iter: {_iter}/{self.total_iter}. Loop: Train. Losses: {stringify_losses(losses)}")
            self.scheduler.step()

            if is_rank_zero(self.rank):
                if self.ema:
                    self.ema.update()

            if self.should_log and self.step % 50 == 0:
                wandb.log({f"Train/{name}": loss.item()
                            for name, loss in losses.items()}, step=self.step)
                
                # log the learning rates
                wandb.log({f"LR/{i}": lr 
                            for i, lr in enumerate(self.scheduler.get_last_lr())}, step=self.step)

            self.step += 1

            # ------------------------------------------------------------------------------------------------------

            if self.test_loader:
                if (self.step % self.cfg.training.val_freq) == 0:
                    self.model.eval()
                    if self.conditioner:
                        self.conditioner.eval()
                    if self.should_write:
                        self.save_checkpoint(
                            f"{self.experiment_id}_latest.pt")

                    # ------------------------------------------------------------------------------------------------------
                    # Validation loop
                    # ------------------------------------------------------------------------------------------------------
                    # validate on the entire validation set in every process but save only from rank 0, I know, inefficient, but avoids divergence of processes
                    metrics, test_losses = self.validate()
                    # print("Validated: {}".format(metrics))
                    if self.should_log:
                        wandb.log(
                            {f"Test/{name}": tloss for name, tloss in test_losses.items()}, step=self.step)

                        wandb.log({f"Metrics/{k}": v for k,
                                    v in metrics.items()}, step=self.step)

                        if (metrics[self.metric_criterion] < best_loss) and self.should_write:
                            self.save_checkpoint(
                                f"{self.experiment_id}_best.pt")
                            best_loss = metrics[self.metric_criterion]

                    self.model.train()
                    if self.conditioner:
                        self.conditioner.train()

                    if self.cfg.num_gpus > 1:
                        dist.barrier()

            # ------------------------------------------------------------------------------------------------------

        # Save / validate at the end
        self.step += 1  # log as final point
        self.model.eval()
        if self.conditioner:
            self.conditioner.eval()
        self.save_checkpoint(f"{self.experiment_id}_latest.pt")
        if self.test_loader:

            # ------------------------------------------------------------------------------------------------------
            # Validation loop at the end of training
            # ------------------------------------------------------------------------------------------------------
            metrics, test_losses = self.validate()
            # print("Validated: {}".format(metrics))
            if self.should_log:
                wandb.log({f"Test/{name}": tloss for name,
                          tloss in test_losses.items()}, step=self.step)
                wandb.log({f"Metrics/{k}": v for k,
                          v in metrics.items()}, step=self.step)

                if (metrics[self.metric_criterion] < best_loss) and self.should_write:
                    self.save_checkpoint(
                        f"{self.experiment_id}_best.pt")
                    best_loss = metrics[self.metric_criterion]

        self.model.train()

    def validate(self):
        with torch.no_grad():
            losses_avg = RunningAverageDict()
            metrics_avg = RunningAverageDict()
            for i, batch in tqdm(enumerate(self.test_loader), 
                                 desc=f"Iter: {self.step}/{self.total_iter}. Loop: Validation", total=len(self.test_loader), disable=not is_rank_zero(self.rank)):
                metrics, losses = self.validate_on_batch(batch, val_step=i)

                if losses:
                    losses_avg.update(losses)
                if metrics:
                    metrics_avg.update(metrics)

            return metrics_avg.get_value(), losses_avg.get_value()

    def save_checkpoint(self, filename):
        if not self.should_write:
            return
        root = self.save_dir
        if not os.path.isdir(root):
            os.makedirs(root)

        fpath = os.path.join(root, filename)
        m = self.model.module if self.cfg.num_gpus > 1 else self.model
        torch.save(
            {
                "model": m.state_dict(),
                "optimizer": self.optimizer.state_dict(),  # TODO : Change to self.optimizer.state_dict() if resume support is needed, currently None to reduce file size
                "step": self.step
            }, fpath)

    # def log_images(self, rgb: Dict[str, list] = {}, depth: Dict[str, list] = {}, scalar_field: Dict[str, list] = {}, prefix="", scalar_cmap="jet", min_depth=None, max_depth=None):
    #     if not self.should_log:
    #         return

    #     if min_depth is None:
    #         try:
    #             min_depth = self.config.min_depth
    #             max_depth = self.config.max_depth
    #         except AttributeError:
    #             min_depth = None
    #             max_depth = None

    #     depth = {k: colorize(v, vmin=min_depth, vmax=max_depth)
    #              for k, v in depth.items()}
    #     scalar_field = {k: colorize(
    #         v, vmin=None, vmax=None, cmap=scalar_cmap) for k, v in scalar_field.items()}
    #     images = {**rgb, **depth, **scalar_field}
    #     wimages = {
    #         prefix+"Predictions": [wandb.Image(v, caption=k) for k, v in images.items()]}
    #     wandb.log(wimages, step=self.step)

    # def log_line_plot(self, data):
    #     if not self.should_log:
    #         return

    #     plt.plot(data)
    #     plt.ylabel("Scale factors")
    #     wandb.log({"Scale factors": wandb.Image(plt)}, step=self.step)
    #     plt.close()

    # def log_bar_plot(self, title, labels, values):
    #     if not self.should_log:
    #         return

    #     data = [[label, val] for (label, val) in zip(labels, values)]
    #     table = wandb.Table(data=data, columns=["label", "value"])
    #     wandb.log({title: wandb.plot.bar(table, "label",
    #               "value", title=title)}, step=self.step)

class Trainer(BaseTrainer):
    pass