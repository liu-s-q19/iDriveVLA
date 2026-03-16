from pytorch_lightning.callbacks import TQDMProgressBar


def build_tqdm_progress_bar(training_cfg, default_refresh_rate: int = 10):
    if not bool(training_cfg.get("enable_progress_bar", True)):
        return None
    refresh_rate = max(1, int(training_cfg.get("progress_bar_refresh_rate", default_refresh_rate)))
    return TQDMProgressBar(refresh_rate=refresh_rate)
