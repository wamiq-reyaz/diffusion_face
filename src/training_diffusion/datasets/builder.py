from .latent import WData

def create_dataset(cfg):
    dataset = WData(
        cfg=cfg,
        w_path=cfg.dataset.w_path,
        img_path=cfg.dataset.img_path,
        stats_path=cfg.dataset.stats_path,
        padding=cfg.dataset.padding,
        image_size=cfg.dataset.image_size,
        normalize_w=cfg.dataset.normalize_w,
        normalize_image=cfg.dataset.normalize_image,
        w_norm_type=cfg.dataset.w_norm_type,
        z_scaler=cfg.dataset.z_scaler
    )

    return dataset