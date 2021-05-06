import argparse
import pytorch_lightning as pl

import torch

from pl_module import get_module


class OnSaveCallback(pl.Callback):
    def on_fit_start(self, trainer, pl_module):
        pl_module.log_dir = trainer.logger.log_dir


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--module_name", type=str, default="base", help="choose model name")
    parser.add_argument("--train", action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--path', type=str)

    temp_args, _ = parser.parse_known_args()
    M = get_module(temp_args.module_name)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = M.add_module_specific_args(parser)
    args = parser.parse_args()

    module = M(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args,
                                            callbacks=[OnSaveCallback(),],
                                            default_root_dir=f'./log/{temp_args.module_name}')

    if temp_args.train:
        if args.auto_lr_find:
            trainer.tune(module)
        trainer.fit(module)

    if temp_args.test:
        if temp_args.path is not None and not temp_args.train:
            state_dict = torch.load(temp_args.path)
            module.load_from_checkpoint(temp_args.path)

        trainer.test(module)

