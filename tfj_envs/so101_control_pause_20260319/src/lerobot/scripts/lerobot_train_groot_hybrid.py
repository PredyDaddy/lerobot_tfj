#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib import import_module

from lerobot.configs import parser
from lerobot.configs.train_groot_hybrid import TrainGrootHybridConfig


def _load_train():
    try:
        trainer_module = import_module("lerobot.rl.groot_hybrid.trainer")
    except ModuleNotFoundError as exc:
        if exc.name != "lerobot.rl.groot_hybrid.trainer":
            raise
        raise ModuleNotFoundError(
            "Groot hybrid training CLI expects `lerobot.rl.groot_hybrid.trainer` "
            "to exist and export `train(cfg)`."
        ) from exc

    try:
        return trainer_module.train
    except AttributeError as exc:
        raise AttributeError(
            "`lerobot.rl.groot_hybrid.trainer` must define `train(cfg)` for this CLI entrypoint."
        ) from exc


@parser.wrap()
def main(cfg: TrainGrootHybridConfig):
    _load_train()(cfg)


if __name__ == "__main__":
    main()
