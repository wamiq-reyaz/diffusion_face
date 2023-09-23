from latent import WData

def create_dataset(cfg):
    dataset = WData(
        w_path=cfg.dataset.w_path,
        img_path=cfg.dataset.img_path,
        stats_path=cfg.dataset.stats_path,
        padding=cfg.dataset.padding,
        image_size=cfg.dataset.image_size,
        normalize_w=cfg.dataset.normalize_w,
        normalize_image=cfg.dataset.normalize_image
    )

    return dataset