




# trainer class

class Trainer1D(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion1D,
        dataset: Dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        conditional_model = None,
        normalize_condition = True,
        conditional_lr= 0.0
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no',
            # kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
        )

        self.accelerator.native_amp = amp

        # model

        self.model = diffusion_model
        self.model.forward = self.model.forward1
        self.channels = diffusion_model.channels
        self.conditional_model = conditional_model
        self.normalize_condition = normalize_condition
        self.conditional_lr = conditional_lr
        if self.model.model.is_conditional:
            assert self.conditional_model, "The diffusion model is conditional but no conditional model exists in the Trainer."
        

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        # dataset and dataloader

        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 8, persistent_workers=True,
                        prefetch_factor=5)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizernormalize_con
        param_groups = [{'params': self.model.parameters(),
                         'lr': train_lr,
                         'betas': adam_betas}]
        if self.conditional_lr:
            param_groups.append({'params': self.conditional_model.parameters(),
                                    'lr': self.conditional_lr,
                                    'betas': adam_betas})
        else:
            pass
        self.opt = Adam(params=param_groups)

        self.scheduler = OneCycleLR(self.opt, max_lr=[k['lr'] for k in param_groups], total_steps=self.train_num_steps,
                                    pct_start=0.02, div_factor=25)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.writer = SummaryWriter(self.results_folder)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        models_list = [self.model, self.conditional_model] #if self.conditional_model else [self.model]
        # self.model, self.conditional_model, self.opt, self.opt_cond, self.scheduler = self.accelerator.prepare(*models_list, self.opt, self.opt_cond, self.scheduler)
        self.model, self.conditional_model, self.opt, self.scheduler = self.accelerator.prepare(*models_list, self.opt, self.scheduler)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'cond_model': self.accelerator.get_state_dict(self.conditional_model) if self.conditional_model else None,
            'opt': self.opt.state_dict(),
            # 'opt_cond': self.opt_cond.state_dict() if self.conditional_model else None,
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, full_path=False):
        accelerator = self.accelerator
        device = accelerator.device

        if full_path:
            data = torch.load(str(milestone), map_location=device)
        else:
            data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])

        if self.conditional_lr:
            try:
                conditional_model = self.accelerator.unwrap_model(self.conditional_model)
                conditional_model.load_state_dict(data['cond_model'])
            except Exception as e:
                print(e)
                print('Failed to load conditional model weights')
            
            try:
                self.opt_cond.load_state_dict(data['opt_cond'])
            except Exception as e:
                print(e)
                print('Failed to load conditional optimizer state')

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])


        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
    
    def load_only_weights(self, milestone, full_path=False):
        accelerator = self.accelerator
        device = accelerator.device

        if full_path:
            data = torch.load(str(milestone), map_location=device)
        else:
            data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    _data = next(self.dl)
                    if self.conditional_model:
                        data = _data['data'].to(device)
                        condition = _data['condition'].to(device)
                        image = condition.clone()
                    else:
                        data = _data.to(device)
                        condition = None

                    with self.accelerator.autocast():
                        if self.conditional_model:
                            _context = nullcontext() if self.conditional_lr else torch.no_grad()
                            with _context:
                                # TODO: why does this need unsqueeze
                                B = condition.shape[0]
                                condition = self.conditional_model(condition).view(B, 512, -1) # BxExS
                                
                                if self.normalize_condition:
                                    condition = torch.tanh(condition)

                        loss = self.model(data, condition=condition)

                        if accelerator.is_main_process:
                            self.writer.add_scalar('loss/train/', loss.item(), self.step)
                            self.writer.add_scalar('lr', self.scheduler.get_lr()[0], self.step)
                        
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                # for k, v in self.conditional_model.module.named_modules():
                #     print(k)
                # print(self.conditional_model.module.conv1.weight.grad)
                # quit()

                accelerator.wait_for_everyone()

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                if self.conditional_model:
                    accelerator.clip_grad_norm_(self.conditional_model.parameters(), 1.0)

                pbar.set_description(f'loss: {total_loss:.4f}')

                self.opt.step()
                self.opt.zero_grad()
                # if self.opt_cond:
                #     self.opt_cond.step()
                #     self.opt_cond.zero_grad()
                self.scheduler.step()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        # TODO: figure out how to distribute the image condition as well
                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n, condition=None if (condition is None) else condition[:n]), batches))
                        #
                        all_samples = torch.cat(all_samples_list, dim = 0)
                        #
                        torch.save(all_samples, str(self.results_folder / f'sample-{milestone}.png'))
                        if not (condition is None):
                            torch.save(image, str(self.results_folder / f'sample_condition-{milestone}.png'))
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')


# trainer class

class TrainerPhotometric(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion1D,
        dataset: Dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        conditional_model = None,
        normalize_condition = True,
        generator_model = None,
        dataset_mean = None,
        dataset_std = None,
        _range = None,
        _min = None,
        camera_params = None,
        v = None,
        photometric_weight = None
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        # model

        self.model = diffusion_model
        self.model.forward = self.model.forward2
        self.channels = diffusion_model.channels
        self.conditional_model = conditional_model
        self.normalize_condition = normalize_condition
        self.generator_model = generator_model
        # photometric/rendering params
        self.dataset_mean = dataset_mean 
        self.dataset_std = dataset_std 
        self._range = _range 
        self._min = _min 
        self.camera_params = camera_params 
        self.v = v 
        self.photometric_weight = photometric_weight

        if self.model.model.is_conditional:
            assert self.conditional_model, "The diffusion model is conditional but no conditional model exists in the Trainer."

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        # dataset and dataloader

        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count()//8)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        self.scheduler = OneCycleLR(self.opt, max_lr=train_lr, total_steps=self.train_num_steps,
                                    pct_start=0.02, div_factor=25)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # Tensorboard logging functionality
        self.writer = SummaryWriter(self.results_folder)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        models_list = [self.model, self.conditional_model] if self.conditional_model else [self.model]
        self.model, self.conditional_model, self.opt, self.scheduler, self.generator_model = self.accelerator.prepare(*models_list, self.opt, self.scheduler, self.generator_model)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, full_path=False):
        accelerator = self.accelerator
        device = accelerator.device

        if full_path:
            data = torch.load(str(milestone), map_location=device)
        else:
            data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
    
    def load_only_weights(self, milestone, full_path=False):
        accelerator = self.accelerator
        device = accelerator.device

        if full_path:
            data = torch.load(str(milestone), map_location=device)
        else:
            data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    _data = next(self.dl)
                    if self.conditional_model:
                        data = _data['data'].to(device)
                        condition = _data['condition'].to(device)
                        image = condition.clone()
                    else:
                        data = _data.to(device)
                        condition = None

                    with self.accelerator.autocast():
                        if self.conditional_model:
                            with torch.no_grad():
                                # TODO: why does this need unsqueeze
                                B = condition.shape[0]
                                condition = self.conditional_model(condition).view(B, 512, -1) # BxExS
                                
                                if self.normalize_condition:
                                    condition = torch.special.expit(condition)

                        loss_diffusion, latents, loss_weight = self.model(data, condition=condition)
                        # # _min, _range and latents are of shape 1xExS
                        # latents = (latents  + 1 ) * 0.5 # normalize to 0, 1
                        # latents = (latents  * self._range.to(device)) + self._min.to(device) # min-max normalization
                        # latents = latents[..., 7:]
                        # latents = latents.permute(0, 2, 1) # BxSxE
                        # all_hooks = set_replacement_hook(generator=self.generator_model,
                        #                                 names=WS,
                        #                                 tensors=latents)
                        
                        # # create random ws for a pass 
                        # b = latents.shape[0]
                        # ws = torch.rand((b, 28, 512)).to(device)
                        # pred_img = self.generator_model.synthesis(ws,
                        #                                      c=torch.repeat_interleave(input=self.camera_params, repeats=b, dim=0).to(device),
                        #                                      v=torch.repeat_interleave(input=self.v, repeats=b, dim=0).to(device),
                        #                                      noise_mode='const')['image']
                        # pred_img = torch.nn.functional.interpolate(pred_img,
                        #                                            size=256,
                        #                                            mode='bilinear',
                        #                                            align_corners=False)
                        # # normalization
                        # pred_img = (pred_img + 1) / 2.0

                        # image = image * self.dataset_std.to(device)
                        # image = image + self.dataset_mean.to(device)

                        # loss_photometric = self.photometric_weight * F.smooth_l1_loss(image, pred_img, reduction='none')
                        # loss_photometric = reduce(loss_photometric, 'b ... -> b (...)', 'mean' )
                        # loss_photometric = (loss_weight*loss_photometric).mean()
                        loss = loss_diffusion # + loss_photometric


                        # perform logging
                        if accelerator.is_main_process:
                            self.writer.add_scalar('loss_diffusion/train/', loss_diffusion.item(), self.step)
                            # self.writer.add_scalar('loss_photometric/train/', loss_photometric.item(), self.step)
                            self.writer.add_scalar('loss/train/', loss.item(), self.step)
                            self.writer.add_scalar('lr', self.scheduler.get_lr()[0], self.step)

                            if (self.step % 1000) == 0: # log images every 1000 steps
                                gt_grid = torchvision.utils.make_grid(image[:4], normalize=True, value_range=(0, 1))
                                pred_grid = torchvision.utils.make_grid(pred_img[:4], normalize=True, value_range=(0, 1))
                                diff_img = torch.abs(gt_grid - pred_grid).detach().cpu().numpy() # BxCxHxW
                                diff_img = np.sum(diff_img, axis=0) # BxHxW 
                                diff_img = colorize(diff_img) # HxWWx3
                                diff_img = torch.from_numpy(diff_img).permute(2, 0, 1)

                                self.writer.add_image('images/train/gt', gt_grid, self.step)
                                self.writer.add_image('images/train/pred', pred_grid, self.step)
                                self.writer.add_image('images/train/l1_diff', diff_img, self.step)
                                
                        for h in all_hooks:
                            h.remove()
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        # # TODO: figure out how to distribute the image condition as well
                        # with torch.no_grad():
                        milestone = self.step // self.save_and_sample_every
                        #     batches = num_to_groups(self.num_samples, self.batch_size)
                        #     all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n, condition=condition[:n]), batches))
                        # #
                        # all_samples = torch.cat(all_samples_list, dim = 0)
                        # #
                        # torch.save(all_samples, str(self.results_folder / f'sample-{milestone}.png'))
                        # if not (condition is None):
                        #     torch.save(image, str(self.results_folder / f'sample_condition-{milestone}.png'))
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
        self.writer.close()