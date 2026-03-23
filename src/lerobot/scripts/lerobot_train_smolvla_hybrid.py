#!/usr/bin/env python

from lerobot.configs import parser
from lerobot.configs.train_smolvla_hybrid import TrainSmolVLAHybridConfig
from lerobot.rl.smolvla_hybrid.trainer import train as train_smolvla_hybrid


@parser.wrap()
def train(cfg: TrainSmolVLAHybridConfig) -> None:
    train_smolvla_hybrid(cfg)


if __name__ == "__main__":
    train()
