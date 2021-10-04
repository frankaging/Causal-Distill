# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team and Facebook, Inc.
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
""" Utils to train DistilBERT
    adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
import json
import logging
import os
import socket

import git
import numpy as np
import torch


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def git_log(folder_path: str):
    """
    Log commit info.
    """
    repo = git.Repo(search_parent_directories=True)
    repo_infos = {
        "repo_id": str(repo),
        "repo_sha": str(repo.head.object.hexsha),
        "repo_branch": str(repo.active_branch),
    }

    with open(os.path.join(folder_path, "git_log.json"), "w") as f:
        json.dump(repo_infos, f, indent=4)


def init_gpu_params(params):
    """
    Handle single and multi-GPU / multi-node.
    """
    if params.n_gpu <= 0:
        params.local_rank = 0
        params.master_port = -1
        params.is_master = True
        params.multi_gpu = False
        return

    assert torch.cuda.is_available()

    logger.info("Initializing GPUs")
    if params.n_gpu > 1:
        # really? comment out.
        # assert params.local_rank != -1
        params.local_rank = -1

        # params.world_size = int(os.environ["WORLD_SIZE"])
        # params.n_gpu_per_node = int(os.environ["N_GPU_NODE"])
        # params.global_rank = int(os.environ["RANK"])

        # number of nodes / node ID
        # params.n_nodes = params.world_size // params.n_gpu_per_node
        # params.node_id = params.global_rank // params.n_gpu_per_node
        params.multi_gpu = True

        # assert params.n_nodes == int(os.environ["N_NODES"])
        # assert params.node_id == int(os.environ["NODE_RANK"])
    params.is_master = True

def set_seed(args):
    """
    Set the random seed.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
