import os
import json
import csv
import datetime
from pathlib import Path
import logging

import tensorflow as tf


class ExperimentRun:

    def __init__(self, params, config=None, base_dir="experiments", run_name=None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        if run_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = f"run_{timestamp}"

        self.run_name = run_name
        self.run_dir = self.base_dir / run_name
        self.run_dir.mkdir(exist_ok=True)

        self.params_path = self.run_dir / "params.json" #raw hyperparams
        self.config_path = self.run_dir / "config.json" #for reproducebility
        self.train_log_path = self.run_dir / "train.log"
        self.losses_path = self.run_dir / "losses.csv" #loss values epoch, step
        self.ckpt_dir = self.run_dir / "checkpoints" #model and checkpoint per epoch
        self.samples_dir = self.run_dir / "results"
        
        self.ckpt_dir.mkdir(exist_ok=True)
        self.samples_dir.mkdir(exist_ok=True)

        with open(self.params_path, "w") as f:
            json.dump(params, f, indent=2, default=str)
        if config is None:
            config = params
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        self.logger = self._setup_logger()
        self._loss_header_written = self.losses_path.exists()
        self.ckpt = None
        self.ckpt_manager = None
        self.logger.info(f"Experiment directory: {self.run_dir}")


    #logging
    def _setup_logger(self):
        logger = logging.getLogger(self.run_name)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            fh = logging.FileHandler(self.train_log_path, mode="a", encoding="utf-8")
            fh.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            fh.setFormatter(fmt)
            ch.setFormatter(fmt)
            logger.addHandler(fh)
            logger.addHandler(ch)
        return logger

    def get_logger(self):
        return self.logger

    #checkpoints
    def create_checkpoint_manager(self, **objects):
        self.ckpt = tf.train.Checkpoint(**objects)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt,
            directory=str(self.ckpt_dir),
            max_to_keep=None  
        )
        return self.ckpt, self.ckpt_manager

    def save_checkpoint(self):
        if self.ckpt_manager is None:
            raise RuntimeError("CheckpointManager not created. Call create_checkpoint_manager first.")
        path = self.ckpt_manager.save()
        self.logger.info(f"Saved checkpoint: {path}")
        return path

    #loss
    def log_losses(self, epoch: int, step: int, loss_dict: dict):
        row = {"epoch": epoch, "step": step}
        for k, v in loss_dict.items():
            try:
                row[k] = float(v.numpy())
            except AttributeError:
                row[k] = float(v)
        file_exists = self.losses_path.exists()
        with open(self.losses_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
