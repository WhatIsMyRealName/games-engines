import math
import os
import time
from typing import TypeAlias

import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence

NDarrayFloat32: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
NDarrayFloat32_1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float32]]
IntLike: TypeAlias = int | np.int64
FloatLike: TypeAlias = float | np.float32

BOARD_SIZE = 5
DIRS = [(-1, 1), (-1, 0), (-1, -1), (0, 1), (0, -1), (1, -1), (1, 0), (1, 1)]
DIR_TO_IDX = {direction: i for i, direction in enumerate(DIRS)}
POLICY_SHAPE = (8, BOARD_SIZE, BOARD_SIZE, 8)
POLICY_SIZE = math.prod(POLICY_SHAPE)

SELF_PLAY_N_SIMULATIONS = 4096
SELF_PLAY_MAX_PLIES = 80
SELF_PLAY_TAU = 0.75
SELF_PLAY_FIRST_MOVE_TAU = 1.0
SELF_PLAY_VALUE_GAMMA = 0.95

MCTS_C_PUCT = 2.0 # 3
MCTS_PUCT_PARENT_EXPONENT = 0.5
MCTS_DIRICHLET_ALPHA_0 = 10.0
MCTS_DIRICHLET_NOISE_WEIGHT = 0.20

TORCH_INFERENCE_THREADS = 1
TORCH_INTEROP_THREADS = 1

RNG = RandomState(MT19937(SeedSequence(int(time.time() * 1e6) + os.getpid())))
