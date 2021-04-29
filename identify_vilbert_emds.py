# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import random
from io import open
import numpy as np

import yaml
from easydict import EasyDict as edict

import torch
import torch.nn as nn

from vilbert.task_utils import (
    LoadLosses,
    LoadDatasetTransfer,
)
from vilbert.vilbert import VILBertForVLTasks, BertForMultiModalPreTraining
from vilbert.vilbert import BertConfig
import torch.distributed as dist

from sklearn.preprocessing import MinMaxScaler


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--from_pretrained",
        default="save/multitask_model/multi_task_model.bin",
        type=str,
        help="VILBert pre-trained model selected in the list: multi_task_model.bin, pretrained_model.bin",
    )

    parser.add_argument(
        "--config_file",
        default="config/bert_base_6layer_6conect.json",
        type=str,
        help="The config file which specified the model details.",
    )

    parser.add_argument(
        "--output_dir",
        default="results",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )

    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )

    parser.add_argument(
        "--tasks", default="8", type=str, help="1-2-3... training task separate by -"
    )
    parser.add_argument(
        "--in_memory",
        default=False,
        type=bool,
        help="whether use chunck for parallel training.",
    )

    parser.add_argument("--batch_size", default=1, type=int, help="which split to use.")

    args = parser.parse_args()
    with open("vilbert_transfer_tasks.yml", "r") as f:
        task_cfg = edict(yaml.safe_load(f))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if "/" in args.from_pretrained:
        timeStamp = args.from_pretrained.split("/")[1]
    else:
        timeStamp = args.from_pretrained

    savePath = os.path.join(args.output_dir, timeStamp)

    config = BertConfig.from_json_file(args.config_file)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")

    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    if default_gpu and not os.path.exists(savePath):
        os.makedirs(savePath)

    task_batch_size, task_num_iters, task_id, task_datasets_val, task_dataloader_val = LoadDatasetTransfer(
        args, task_cfg, args.tasks
    )

    num_labels = max([dataset.num_labels for dataset in task_datasets_val.values()])

    config.fast_mode = True

    model = VILBertForVLTasks.from_pretrained(
        args.from_pretrained,
        config=config,
        num_labels=num_labels,
        default_gpu=default_gpu,
    )

    torch.cuda.empty_cache()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, deay_allreduce=True)

    elif n_gpu > 1:
        model = nn.DataParallel(model)

    print("***** Running feature extraction *****")
    print("  Batch size: ", task_batch_size)

    model.eval()
    image_ids = []

    nocap = len(task_datasets_val[task_id]._caption_entries)
    feature_v_matrix = np.zeros((nocap, 1024))
    feature_t_matrix = np.zeros((nocap, 768))

    for i, batch in enumerate(task_dataloader_val[task_id]):
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
        features, spatials, image_mask, question, input_mask, segment_ids, caption_idx, image_idx = (
            batch
        )

        image_ids.append(image_idx.item())

        task_tokens = (
            question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))
        )

        with torch.no_grad():
            seq_t_output, seq_v_output = model.getembedding(
                question,
                features,
                spatials,
                segment_ids,
                input_mask,
                image_mask,
                task_ids=task_tokens,
            )

            feature_v_matrix[caption_idx, :] = (seq_v_output.view(-1).cpu().numpy())
            feature_t_matrix[caption_idx, :] = (seq_t_output.view(-1).cpu().numpy())

        np.save(args.output_dir + '/flickr30k_trans_embeddings/flickr30k_vilbert_features_imgs', feature_v_matrix)
        np.save(args.output_dir + '/flickr30k_trans_embeddings/flickr30k_vilbert_features_caps', feature_t_matrix)
        with open(args.output_dir + '/flickr30k_trans_embeddings/img_path.txt', 'w') as f:
            for item in image_ids:
                f.write("%s\n" % item)


if __name__ == "__main__":
    main()
