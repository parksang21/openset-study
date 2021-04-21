import argparse
import pytorch_lightning as pl

from pl_module import get_module


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--module_name", type=str, default="base", help="choose model name")

    temp_args, _ = parser.parse_known_args()
    M = get_module(temp_args.module_name)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = M.add_module_specific_args(parser)
    args = parser.parse_args()

    module = M(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(module)
