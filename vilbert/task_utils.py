# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from io import open
import json
import logging
import os
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers.tokenization_bert import BertTokenizer
from vilbert.datasets import DatasetMapTrain, DatasetMapEval, DatasetMapTrans
from vilbert.datasets._image_features_reader import ImageFeaturesH5Reader
import pdb

logger = logging.getLogger(__name__)

LossMap = {
    "BCEWithLogitLoss": nn.BCEWithLogitsLoss(reduction="mean"),
    "CrossEntropyLoss": nn.CrossEntropyLoss(),
}


def ForwardModelsVal(args, task_cfg, device, task_id, batch, model, task_losses):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

    features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = (
        batch
    )

    batch_size = features.size(0)

    max_num_bbox = features.size(1)
    num_options = question.size(1)
    features = features.view(-1, features.size(2), features.size(3))
    spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
    image_mask = image_mask.view(-1, image_mask.size(2))
    question = question.view(-1, question.size(2))
    input_mask = input_mask.view(-1, input_mask.size(2))
    segment_ids = segment_ids.view(-1, segment_ids.size(2))
    co_attention_mask = co_attention_mask.view(
        -1, co_attention_mask.size(2), co_attention_mask.size(3)
    )

    task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))

    vil_logit = model(
        question,
        features,
        spatials,
        segment_ids,
        input_mask,
        image_mask,
        co_attention_mask,
        task_tokens,
    )

    vil_logit = vil_logit.view(batch_size, num_options)
    loss = task_losses[task_id](vil_logit, target)
    _, preds = torch.max(vil_logit, 1)
    batch_score = (preds == target).sum()

    return float(loss), float(batch_score), batch_size


def ForwardModelsTrain(
    args,
    task_cfg,
    device,
    task_id,
    task_count,
    task_iter_train,
    task_dataloader_train,
    model,
    task_losses,
):
    # given the current task, decided whether to forward the model and forward with specific loss.

    # reset the task iteration when needed.
    if task_count[task_id] % len(task_dataloader_train[task_id]) == 0:
        task_iter_train[task_id] = iter(task_dataloader_train[task_id])

    task_count[task_id] += 1
    # get the batch
    batch = task_iter_train[task_id].next()
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

    features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = (
        batch
    )

    batch_size = features.size(0)

    max_num_bbox = features.size(1)
    num_options = question.size(1)
    features = features.view(-1, features.size(2), features.size(3))
    spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
    image_mask = image_mask.view(-1, image_mask.size(2))
    question = question.view(-1, question.size(2))
    input_mask = input_mask.view(-1, input_mask.size(2))
    segment_ids = segment_ids.view(-1, segment_ids.size(2))
    co_attention_mask = co_attention_mask.view(
        -1, co_attention_mask.size(2), co_attention_mask.size(3)
    )

    task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))
    vil_logit = model(
        question,
        features,
        spatials,
        segment_ids,
        input_mask,
        image_mask,
        co_attention_mask,
        task_tokens,
    )

    vil_logit = vil_logit.view(batch_size, num_options)
    loss = task_losses[task_id](vil_logit, target)
    _, preds = torch.max(vil_logit, 1)
    batch_score = float((preds == target).sum()) / float(batch_size)

    return loss, batch_score


def LoadLosses(args, task_cfg, task_ids):
    losses = {}
    task_types = []
    num_labels = 0
    for i, task_id in enumerate(task_ids):
        task = "TASK" + task_id
        model_type = task_cfg[task]["type"]
        if model_type not in task_types:
            task_types.append(model_type)
        losses[task] = LossMap[task_cfg[task]["loss"]]

    return losses


def LoadDatasets(args, task_cfg, task_id, split="trainval"):

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    task_feature_reader1 = {}
    task_feature_reader2 = {}

    task = "TASK" + task_id
    if task_cfg[task]["features_h5path1"] not in task_feature_reader1:
        task_feature_reader1[task_cfg[task]["features_h5path1"]] = None
    if task_cfg[task]["features_h5path2"] not in task_feature_reader2:
        task_feature_reader2[task_cfg[task]["features_h5path2"]] = None

    # initilzie the feature reader
    for features_h5path in task_feature_reader1.keys():
        if features_h5path != "":
            task_feature_reader1[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )
    for features_h5path in task_feature_reader2.keys():
        if features_h5path != "":
            task_feature_reader2[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )

    task_datasets_train = {}
    task_datasets_val = {}
    task_dataloader_train = {}
    task_dataloader_val = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}

    task = "TASK" + task_id
    task_name = task_cfg[task]["name"]
    task_ids.append(task)
    batch_size = task_cfg[task]["batch_size"] // args.gradient_accumulation_steps
    num_workers = args.num_workers
    if args.local_rank != -1:
        batch_size = int(batch_size / dist.get_world_size())
        num_workers = int(num_workers / dist.get_world_size())

    # num_workers = int(num_workers / len(ids))
    logger.info(
        "Loading %s Dataset with batch size %d"
        % (task_cfg[task]["name"], batch_size)
    )

    task_datasets_train[task] = None
    if "train" in split:
        task_datasets_train[task] = DatasetMapTrain[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["train_annotations_jsonpath"],
            split=task_cfg[task]["train_split"],
            image_features_reader=task_feature_reader1[
                task_cfg[task]["features_h5path1"]
            ],
            gt_image_features_reader=task_feature_reader2[
                task_cfg[task]["features_h5path2"]
            ],
            tokenizer=tokenizer,
            bert_model=args.bert_model,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
        )

    task_datasets_val[task] = None
    if "val" in split:
        task_datasets_val[task] = DatasetMapTrain[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
            split=task_cfg[task]["val_split"],
            image_features_reader=task_feature_reader1[
                task_cfg[task]["features_h5path1"]
            ],
            gt_image_features_reader=task_feature_reader2[
                task_cfg[task]["features_h5path2"]
            ],
            tokenizer=tokenizer,
            bert_model=args.bert_model,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
        )

    task_num_iters[task] = 0
    task_batch_size[task] = 0
    if "train" in split:
        if args.local_rank == -1:
            train_sampler = RandomSampler(task_datasets_train[task])
        else:
            # TODO: check if this works with current data generator from disk that relies on next(file)
            # (it doesn't return item back by index)
            train_sampler = DistributedSampler(task_datasets_train[task])

        task_dataloader_train[task] = DataLoader(
            task_datasets_train[task],
            sampler=train_sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        task_num_iters[task] = len(task_dataloader_train[task])
        task_batch_size[task] = batch_size

    if "val" in split:
        task_dataloader_val[task] = DataLoader(
            task_datasets_val[task],
            shuffle=False,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
        )

    return (
        task_batch_size,
        task_num_iters,
        task_ids,
        task_datasets_train,
        task_datasets_val,
        task_dataloader_train,
        task_dataloader_val,
    )


def LoadDatasetEval(args, task_cfg, ids):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    task_feature_reader1 = {}
    task_feature_reader2 = {}
    for i, task_id in enumerate(ids):
        task = "TASK" + task_id
        if task_cfg[task]["features_h5path1"] not in task_feature_reader1:
            task_feature_reader1[task_cfg[task]["features_h5path1"]] = None
        if task_cfg[task]["features_h5path2"] not in task_feature_reader2:
            task_feature_reader2[task_cfg[task]["features_h5path2"]] = None

    # initilzie the feature reader
    for features_h5path in task_feature_reader1.keys():
        if features_h5path != "":
            task_feature_reader1[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )

    for features_h5path in task_feature_reader2.keys():
        if features_h5path != "":
            task_feature_reader2[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )

    task_datasets_val = {}
    task_dataloader_val = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}

    for i, task_id in enumerate(ids):
        task = "TASK" + task_id
        task_ids.append(task)
        task_name = task_cfg[task]["name"]
        batch_size = args.batch_size
        if args.local_rank != -1:
            batch_size = int(batch_size / dist.get_world_size())

        num_workers = int(args.num_workers / len(ids))
        logger.info(
            "Loading %s Dataset with batch size %d"
            % (task_cfg[task]["name"], batch_size)
        )

        if args.split:
            eval_split = args.split
        else:
            eval_split = task_cfg[task]["val_split"]

        task_datasets_val[task] = DatasetMapEval[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
            split=eval_split,
            image_features_reader=task_feature_reader1[
                task_cfg[task]["features_h5path1"]
            ],
            gt_image_features_reader=task_feature_reader2[
                task_cfg[task]["features_h5path2"]
            ],
            tokenizer=tokenizer,
            bert_model=args.bert_model,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
        )

        task_dataloader_val[task] = DataLoader(
            task_datasets_val[task],
            shuffle=False,
            batch_size=batch_size,
            #num_workers=10,
            pin_memory=True,
        )

        task_num_iters[task] = len(task_dataloader_val[task])
        task_batch_size[task] = batch_size

    return (
        task_batch_size,
        task_num_iters,
        task_ids,
        task_datasets_val,
        task_dataloader_val,
    )


def LoadDatasetTransfer(args, task_cfg, id):

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    task_feature_reader1 = {}
    task_feature_reader2 = {}

    task = "TASK" + id
    if task_cfg[task]["features_h5path1"] not in task_feature_reader1:
        task_feature_reader1[task_cfg[task]["features_h5path1"]] = None
    if task_cfg[task]["features_h5path2"] not in task_feature_reader2:
        task_feature_reader2[task_cfg[task]["features_h5path2"]] = None

    # initilzie the feature reader
    for features_h5path in task_feature_reader1.keys():
        if features_h5path != "":
            task_feature_reader1[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )

    for features_h5path in task_feature_reader2.keys():
        if features_h5path != "":
            task_feature_reader2[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )

    task_datasets_val = {}
    task_dataloader_val = {}
    task_batch_size = {}
    task_num_iters = {}

    task_ids = task
    task_name = task_cfg[task]["name"]
    batch_size = args.batch_size
    if args.local_rank != -1:
        batch_size = int(batch_size / dist.get_world_size())

    logger.info(
        "Loading %s Dataset with batch size %d"
        % (task_cfg[task]["name"], batch_size)
    )

    task_datasets_val[task] = DatasetMapTrans[task_name](
        task=task_cfg[task]["name"],
        dataroot=task_cfg[task]["dataroot"],
        annotations_jsonpath=task_cfg[task]["trans_annotations_jsonpath"],
        image_features_reader=task_feature_reader1[
            task_cfg[task]["features_h5path1"]
        ],
        gt_image_features_reader=task_feature_reader2[
            task_cfg[task]["features_h5path2"]
        ],
        tokenizer=tokenizer,
        bert_model=args.bert_model,
        padding_index=0,
        max_seq_length=task_cfg[task]["max_seq_length"],
        max_region_num=task_cfg[task]["max_region_num"],
    )

    task_dataloader_val[task] = DataLoader(
        task_datasets_val[task],
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
    )

    task_num_iters[task] = len(task_dataloader_val[task])
    task_batch_size[task] = batch_size

    return (
        task_batch_size,
        task_num_iters,
        task_ids,
        task_datasets_val,
        task_dataloader_val,
    )


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores
