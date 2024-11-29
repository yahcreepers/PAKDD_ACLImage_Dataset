import torch
import numpy as np
import os
from argparse import ArgumentParser
from datasets import ALDataModule
from libcll.models import build_model
from libcll.strategies import build_strategy
from strategies import Ordinary
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.cli import LightningCLI

def main(args):
    al_data_module = ALDataModule(args)
    al_data_module.prepare_data()
    al_data_module.setup(stage="fit")
    input_dim, num_classes = al_data_module.train_set.input_dim, al_data_module.train_set.num_classes
    Q, class_priors = al_data_module.get_distribution_info()
    print(Q, class_priors)
    if args.cleaning:
        exit()
        # pass

    pl.seed_everything(args.seed, workers=True)
    
    model = build_model(
        args.model,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
    )

    if args.strategy == "Ord":
        strategy = Ordinary(
            model=model,
            valid_type=args.valid_type,
            num_classes=num_classes,
            type=args.type,
            lr=args.lr,
            Q=Q,
            class_priors=class_priors,
        )
    else:
        strategy = build_strategy(
            args.strategy,
            model=model,
            valid_type=args.valid_type,
            num_classes=num_classes,
            type=args.type,
            lr=args.lr,
            Q=Q,
            class_priors=class_priors,
        )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.output_dir)
    checkpoint_callback_best = ModelCheckpoint(
        monitor=f"Valid_{args.valid_type}",
        dirpath=args.output_dir,
        filename=f"{{epoch}}-{{Valid_{args.valid_type}:.2f}}",
        save_top_k=1,
        mode="max" if args.valid_type == "Accuracy" else "min",
        every_n_epochs=args.eval_epoch,
    )
    checkpoint_callback_last = ModelCheckpoint(
        monitor=f"step",
        dirpath=args.output_dir,
        filename="{epoch}-{step}",
        save_top_k=1,
        mode="max",
        every_n_epochs=args.eval_epoch,
    )

    print("Start Training......")
    trainer = pl.Trainer(
        max_epochs=args.epoch,
        accelerator="gpu",
        logger=tb_logger,
        log_every_n_steps=args.log_step,
        deterministic=True,
        check_val_every_n_epoch=args.eval_epoch,
        callbacks=[checkpoint_callback_best, checkpoint_callback_last],
    )
    trainer.fit(
        strategy,
        datamodule=al_data_module,
    )
    trainer.test(
        datamodule=al_data_module,
        ckpt_path="best"
    )

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="dense")
    parser.add_argument(
        "--model_path", type=str, default=None
    )
    parser.add_argument(
        "--valid_type", type=str, default="URE"
    )
    parser.add_argument(
        "--valid_split", type=float, default=0.1
    )
    parser.add_argument(
        "--eval_epoch", type=int, default=10
    )
    parser.add_argument(
        "--output_dir", type=str, default=None
    )
    parser.add_argument(
        "--batch_size", type=int, default=256
    )
    parser.add_argument("--hidden_dim", type=int, default=500)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--strategy", type=str, default="SCL")
    parser.add_argument("--type", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1126)
    parser.add_argument("--log_step", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--long_label", action="store_true")
    parser.add_argument("--do_transform", action="store_true")
    parser.add_argument("--label_path", type=str, default="cifar10")
    parser.add_argument("--num_cl", type=int, default=3)
    parser.add_argument("--cleaning_rate", type=float, default=0)
    parser.add_argument("--cleaning", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
