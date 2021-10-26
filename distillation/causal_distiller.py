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
""" The distiller to distil the student.
    Adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
import math
import os
import time

import psutil
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups
from lm_seqs_dataset import LmSeqsDataset
from transformers import get_linear_schedule_with_warmup
from utils import logger

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import argparse
import json
import os
import pickle
import shutil
import random

import numpy as np
import torch

from distiller import Distiller
from lm_seqs_dataset import LmSeqsDataset
from transformers import (
    AutoConfig,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    AutoTokenizer,
    AutoModelForMaskedLM,
)
from utils import git_log, init_gpu_params, logger, set_seed
from datasets import load_dataset
from counterfactual_utils import *
import wandb
from models.modeling_distilbert import DistilBertForMaskedLM

# Examples of interchange.
# activations_counterfactual_teacher = get_activation_at(
#     teacher_bert,
#     batch["input_ids"],
#     batch["attention_mask"],
#     variable_names=["$L:1$H:1$[0:32]"]
# )
# interchange_with_activation_at(
#     teacher_bert,
#     batch["input_ids"],
#     batch["attention_mask"],
#     interchanged_variables=[torch.zeros(32, 512, 32)],
#     variable_names=["$L:1$H:1$[0:32]"]
# )

class CausalDistiller:
    def __init__(
        self, params: dict, dataset: LmSeqsDataset, 
        token_probs: torch.tensor, student: nn.Module, teacher: nn.Module
    ):
        if params.is_wandb:
            run = wandb.init(
                project="Causal-BERT-Distillation", 
                entity="wuzhengx",
                name=params.run_name,
            )
            wandb.config.update(params)
        self.is_wandb = params.is_wandb
        
        logger.info("Initializing Distiller")
        self.params = params
        self.dump_path = params.dump_path
        self.multi_gpu = params.multi_gpu
        self.fp16 = params.fp16

        self.student = student
        self.teacher = teacher
        
        # causal neuron mappings.
        self.deserialized_interchange_variable_mappings = []
        with open(params.neuron_mapping) as json_file:
            neuron_mapping = json.load(json_file)
            logger.info(f"Neuron Mapping: {neuron_mapping}")
            interchange_variable_mappings = neuron_mapping["interchange_variable_mappings"]
            for m in interchange_variable_mappings:
                teacher_deserialized_variables = []
                for variable in m["teacher_variable_names"]:
                    teacher_deserialized_variables.append(deserialize_variable_name(variable))
                student_deserialized_variables = []
                for variable in m["student_variable_names"]:
                    student_deserialized_variables.append(deserialize_variable_name(variable))
                self.deserialized_interchange_variable_mappings += [
                    [teacher_deserialized_variables, student_deserialized_variables]
                ]

        self.student_config = student.config
        self.vocab_size = student.config.vocab_size

        # overwrite slightly on this.
        if params.local_rank == -1:
            sampler = RandomSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)
            
        if params.group_by_size:
            groups = create_lengths_groups(lengths=dataset.lengths, k=params.max_model_input_size)
            sampler = GroupedBatchSampler(sampler=sampler, group_ids=groups, batch_size=params.batch_size)
        else:
            sampler = BatchSampler(sampler=sampler, batch_size=params.batch_size, drop_last=False)

        self.dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=dataset.batch_sequences)

        self.temperature = params.temperature
        assert self.temperature > 0.0

        self.alpha_ce = params.alpha_ce
        self.alpha_mlm = params.alpha_mlm
        self.alpha_clm = params.alpha_clm
        self.alpha_mse = params.alpha_mse
        self.alpha_cos = params.alpha_cos
        self.alpha_causal = params.alpha_causal

        self.mlm = params.mlm
        if self.mlm:
            logger.info("Using MLM loss for LM step.")
            self.mlm_mask_prop = params.mlm_mask_prop
            assert 0.0 <= self.mlm_mask_prop <= 1.0
            assert params.word_mask + params.word_keep + params.word_rand == 1.0
            self.pred_probs = torch.FloatTensor([params.word_mask, params.word_keep, params.word_rand])
            self.pred_probs = self.pred_probs.to(torch.device("cuda")) if params.n_gpu > 0 else self.pred_probs
            self.token_probs = token_probs.to(torch.device("cuda")) if params.n_gpu > 0 else token_probs
            if self.fp16:
                self.pred_probs = self.pred_probs.half()
                self.token_probs = self.token_probs.half()
        else:
            logger.info("Using CLM loss for LM step.")

        self.interchange_mlm = params.interchange_mlm
        self.interchange_prop = params.interchange_prop
            
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.last_loss_ce = 0
        self.last_loss_mlm = 0
        self.last_loss_clm = 0
        if self.alpha_mse > 0.0:
            self.last_loss_mse = 0
        if self.alpha_cos > 0.0:
            self.last_loss_cos = 0

        self.last_loss_causal_ce = 0
        self.last_teacher_interchange_efficacy = 0
        self.last_student_interchange_efficacy = 0
        self.last_log = 0

        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        if self.alpha_mse > 0.0:
            self.mse_loss_fct = nn.MSELoss(reduction="sum")
        if self.alpha_cos > 0.0:
            self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

        logger.info("--- Initializing model optimizer")
        assert params.gradient_accumulation_steps >= 1
        self.num_steps_epoch = len(self.dataloader)
        num_train_optimization_steps = (
            int(self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epoch) + 1
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": params.weight_decay,
            },
            {
                "params": [
                    p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        logger.info(
            "------ Number of trainable parameters (student): %i"
            % sum([p.numel() for p in self.student.parameters() if p.requires_grad])
        )
        logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.student.parameters()]))
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=params.learning_rate, eps=params.adam_epsilon, betas=(0.9, 0.98)
        )

        warmup_steps = math.ceil(num_train_optimization_steps * params.warmup_prop)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
        )

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            logger.info(f"Using fp16 training: {self.params.fp16_opt_level} level")
            self.student, self.optimizer = amp.initialize(
                self.student, self.optimizer, opt_level=self.params.fp16_opt_level
            )
            self.teacher = self.teacher.half()

        if self.multi_gpu:
            if self.fp16:
                from apex.parallel import DistributedDataParallel

                logger.info("Using apex.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(self.student)
            else:
                if params.local_rank == -1:
                    logger.info("Using nn.DataParallel for the teacher model.")
                    # teacher also use multi-GPU.
                    self.teacher = torch.nn.DataParallel(self.teacher)
                    self.teacher.to(torch.device("cuda")) # no rank is needed!

                    logger.info("Using nn.DataParallel for the student model.")
                    self.student = torch.nn.DataParallel(self.student)
                    self.student.to(torch.device("cuda")) # no rank is needed!
                else:
                
                    from torch.nn.parallel import DistributedDataParallel

                    logger.info("Using nn.parallel.DistributedDataParallel for distributed training.")
                    self.student = DistributedDataParallel(
                        self.student,
                        device_ids=[params.local_rank],
                        output_device=params.local_rank,
                        find_unused_parameters=True,
                    )

        self.is_master = params.is_master
        
        
    def prepare_batch_mlm(self, batch):
        """
        Prepare the batch: from the token_ids and the lengths, compute the attention mask and the masked label for MLM.

        Input:
        ------
            batch: `Tuple`
                token_ids: `torch.tensor(bs, seq_length)` - The token ids for each of the sequence. It is padded.
                lengths: `torch.tensor(bs)` - The lengths of each of the sequences in the batch.

        Output:
        -------
            token_ids: `torch.tensor(bs, seq_length)` - The token ids after the modifications for MLM.
            attn_mask: `torch.tensor(bs, seq_length)` - The attention mask for the self-attention.
            mlm_labels: `torch.tensor(bs, seq_length)` - The masked language modeling labels. There is a -100 where there is nothing to predict.
        """
        token_ids, lengths = batch
        token_ids, lengths = self.round_batch(x=token_ids, lengths=lengths)
        assert token_ids.size(0) == lengths.size(0)

        attn_mask = torch.arange(token_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None]

        bs, max_seq_len = token_ids.size()
        mlm_labels = token_ids.new(token_ids.size()).copy_(token_ids)

        x_prob = self.token_probs[token_ids.flatten()]
        n_tgt = math.ceil(self.mlm_mask_prop * lengths.sum().item())
        tgt_ids = torch.multinomial(x_prob / x_prob.sum(), n_tgt, replacement=False)
        pred_mask = torch.zeros(
            bs * max_seq_len, dtype=torch.bool, device=token_ids.device
        )  # previously `dtype=torch.uint8`, cf pytorch 1.2.0 compatibility
        pred_mask[tgt_ids] = 1
        pred_mask = pred_mask.view(bs, max_seq_len)

        pred_mask[token_ids == self.params.special_tok_ids["pad_token"]] = 0

        # mask a number of words == 0 [8] (faster with fp16)
        if self.fp16:
            n1 = pred_mask.sum().item()
            if n1 > 8:
                pred_mask = pred_mask.view(-1)
                n2 = max(n1 % 8, 8 * (n1 // 8))
                if n2 != n1:
                    pred_mask[torch.nonzero(pred_mask).view(-1)[: n1 - n2]] = 0
                pred_mask = pred_mask.view(bs, max_seq_len)
                assert pred_mask.sum().item() % 8 == 0, pred_mask.sum().item()

        _token_ids_real = token_ids[pred_mask]
        _token_ids_rand = _token_ids_real.clone().random_(self.vocab_size)
        _token_ids_mask = _token_ids_real.clone().fill_(self.params.special_tok_ids["mask_token"])
        probs = torch.multinomial(self.pred_probs, len(_token_ids_real), replacement=True).to(_token_ids_real.device)
        _token_ids = (
            _token_ids_mask * (probs == 0).long()
            + _token_ids_real * (probs == 1).long()
            + _token_ids_rand * (probs == 2).long()
        )
        token_ids = token_ids.masked_scatter(pred_mask, _token_ids)

        mlm_labels[~pred_mask] = -100  # previously `mlm_labels[1-pred_mask] = -1`, cf pytorch 1.2.0 compatibility

        # sanity checks
        assert 0 <= token_ids.min() <= token_ids.max() < self.vocab_size

        return token_ids, attn_mask, mlm_labels

    def prepare_batch_clm(self, batch):
        """
        Prepare the batch: from the token_ids and the lengths, compute the attention mask and the labels for CLM.

        Input:
        ------
            batch: `Tuple`
                token_ids: `torch.tensor(bs, seq_length)` - The token ids for each of the sequence. It is padded.
                lengths: `torch.tensor(bs)` - The lengths of each of the sequences in the batch.

        Output:
        -------
            token_ids: `torch.tensor(bs, seq_length)` - The token ids after the modifications for MLM.
            attn_mask: `torch.tensor(bs, seq_length)` - The attention mask for the self-attention.
            clm_labels: `torch.tensor(bs, seq_length)` - The causal language modeling labels. There is a -100 where there is nothing to predict.
        """
        token_ids, lengths = batch
        token_ids, lengths = self.round_batch(x=token_ids, lengths=lengths)
        assert token_ids.size(0) == lengths.size(0)

        attn_mask = torch.arange(token_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None]
        clm_labels = token_ids.new(token_ids.size()).copy_(token_ids)
        clm_labels[~attn_mask] = -100  # previously `clm_labels[1-attn_mask] = -1`, cf pytorch 1.2.0 compatibility

        # sanity checks
        assert 0 <= token_ids.min() <= token_ids.max() < self.vocab_size

        return token_ids, attn_mask, clm_labels

    def round_batch(self, x: torch.tensor, lengths: torch.tensor):
        """
        For float16 only.
        Sub-sample sentences in a batch, and add padding, so that each dimension is a multiple of 8.

        Input:
        ------
            x: `torch.tensor(bs, seq_length)` - The token ids.
            lengths: `torch.tensor(bs, seq_length)` - The lengths of each of the sequence in the batch.

        Output:
        -------
            x:  `torch.tensor(new_bs, new_seq_length)` - The updated token ids.
            lengths: `torch.tensor(new_bs, new_seq_length)` - The updated lengths.
        """
        if not self.fp16 or len(lengths) < 8:
            return x, lengths

        # number of sentences == 0 [8]
        bs1 = len(lengths)
        bs2 = 8 * (bs1 // 8)
        assert bs2 > 0 and bs2 % 8 == 0
        if bs1 != bs2:
            idx = torch.randperm(bs1)[:bs2]
            lengths = lengths[idx]
            slen = lengths.max().item()
            x = x[idx, :slen]
        else:
            idx = None

        # sequence length == 0 [8]
        ml1 = x.size(1)
        if ml1 % 8 != 0:
            pad = 8 - (ml1 % 8)
            ml2 = ml1 + pad
            if self.mlm:
                pad_id = self.params.special_tok_ids["pad_token"]
            else:
                pad_id = self.params.special_tok_ids["unk_token"]
            padding_tensor = torch.zeros(bs2, pad, dtype=torch.long, device=x.device).fill_(pad_id)
            x = torch.cat([x, padding_tensor], 1)
            assert x.size() == (bs2, ml2)

        assert x.size(0) % 8 == 0
        assert x.size(1) % 8 == 0
        return x, lengths

    def prepare_interchange_position(self, lengths, dual_lengths):
        interchange_prop = self.interchange_prop
        batch_size = lengths.shape[0]
        interchange_position = []
        for i in range(0, batch_size):
            min_len = min(lengths[i].tolist(), dual_lengths[i].tolist())
            interchange_count = int(min_len*interchange_prop)
            start_index = random.randint(0, lengths[i].tolist()-interchange_count)
            end_index = start_index + interchange_count
            dual_start_index = random.randint(0, dual_lengths[i].tolist()-interchange_count)
            dual_end_index = dual_start_index + interchange_count
            interchange_position += [[start_index, end_index, dual_start_index, dual_end_index]]
        interchange_position = torch.tensor(interchange_position, dtype=torch.long).to(lengths.device)
        return interchange_position
    
    def train(self):
        """
        The real training loop.
        """
        if self.is_master:
            logger.info("Starting training")
        self.last_log = time.time()
        self.student.train()
        self.teacher.eval()

        for _ in range(self.params.n_epoch):
            if self.is_master:
                logger.info(f"--- Starting epoch {self.epoch}/{self.params.n_epoch-1}")

            iter_bar = tqdm(self.dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
            for batch in iter_bar:
                token_ids, lengths, dual_token_ids, dual_lengths = batch
                # from length, let us get the intervention points?
                sampled_interchange_position = self.prepare_interchange_position(lengths, dual_lengths)
                
                if self.params.n_gpu > 0:
                    token_ids = token_ids.to(torch.device("cuda"))
                    lengths = lengths.to(torch.device("cuda"))
                    dual_token_ids = dual_token_ids.to(torch.device("cuda"))
                    dual_lengths = dual_lengths.to(torch.device("cuda"))
                
                if self.mlm:
                    token_ids, attn_mask, lm_labels = self.prepare_batch_mlm(batch=(token_ids, lengths))
                    dual_token_ids, dual_attn_mask, dual_lm_labels = self.prepare_batch_mlm(
                        batch=(dual_token_ids, dual_lengths)
                    )
                else:
                    token_ids, attn_mask, lm_labels = self.prepare_batch_clm(batch=(token_ids, lengths))
                    dual_token_ids, dual_attn_mask, dual_lm_labels = self.prepare_batch_clm(
                        batch=(dual_token_ids, dual_lengths)
                    )
                self.step(
                    input_ids=token_ids, 
                    attention_mask=attn_mask, 
                    lm_labels=lm_labels,
                    dual_input_ids=dual_token_ids, 
                    dual_attention_mask=dual_attn_mask, 
                    dual_lm_labels=dual_lm_labels,
                    sampled_interchange_position=sampled_interchange_position,
                    is_parallel=self.params.parallel_crossway,
                    is_crossway=self.params.include_crossway,
                )
                iter_bar.update()
                iter_bar.set_postfix(
                    {
                        "Last_loss": f"{self.last_loss:.2f}", 
                         "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}", 
                         "Last_cf_loss": f"{self.last_loss_causal_ce:.2f}", 
                    }
                )
            iter_bar.close()

            if self.is_master:
                logger.info(f"--- Ending epoch {self.epoch}/{self.params.n_epoch-1}")
            self.end_epoch()

        if self.is_master:
            logger.info("Save very last checkpoint as `pytorch_model.bin`.")
            self.save_checkpoint(checkpoint_name="pytorch_model.bin")
            logger.info("Training is finished")

    def step(
        self, input_ids: torch.tensor, 
        attention_mask: torch.tensor, 
        lm_labels: torch.tensor,
        dual_input_ids: torch.tensor, 
        dual_attention_mask: torch.tensor, 
        dual_lm_labels: torch.tensor,
        sampled_interchange_position: torch.tensor,
        is_parallel=False,
        is_crossway=False,
    ):
        if is_parallel:
            assert is_crossway
            """
            If we enable crossway and parallel, we make 
            sure we compute pair-wise losses for two 
            examples together.
            Note that this requires larger GPUs.
            """
            self._step_parallel(
                input_ids,
                attention_mask,
                lm_labels,
                dual_input_ids,
                dual_attention_mask,
                dual_lm_labels,
                sampled_interchange_position,
            )
        else:
            """
            If it is not parallel, we will have two mini-step
            within each step. The second step will only backprop
            loss without updating the iteration, so the optimization
            is not affected.
            """
            if is_crossway:
                self._step(
                    input_ids,
                    attention_mask,
                    lm_labels,
                    dual_input_ids,
                    dual_attention_mask,
                    dual_lm_labels,
                    sampled_interchange_position,
                    skip_update_iter=False,
                )
                # the second mini-step for the reversed pair.
                self._step(
                    dual_input_ids,
                    dual_attention_mask,
                    dual_lm_labels,
                    input_ids,
                    attention_mask,
                    lm_labels,
                    sampled_interchange_position,
                    skip_update_iter=True,
                )
            else:
                """
                This subroutine will be the normal distillation
                with optional causal loss.
                """
                self._step(
                    input_ids,
                    attention_mask,
                    lm_labels,
                    dual_input_ids,
                    dual_attention_mask,
                    dual_lm_labels,
                    sampled_interchange_position,
                    skip_update_iter=False,
                )

    def _step(
        self, input_ids: torch.tensor, 
        attention_mask: torch.tensor, 
        lm_labels: torch.tensor,
        dual_input_ids: torch.tensor, 
        dual_attention_mask: torch.tensor, 
        dual_lm_labels: torch.tensor,
        sampled_interchange_position: torch.tensor,
        skip_update_iter=False,
    ):
        """
        One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).

        Input:
        ------
        input_ids/dual_input_ids: `torch.tensor(bs, seq_length)` - The token ids.
        attention_mask/dual_attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
        lm_labels/dual_lm_labels: `torch.tensor(bs, seq_length)` - The language modeling labels (mlm labels for MLM and clm labels for CLM).
        """
        # preparing for causal distillation.
        # we randomly select the pool of neurons to interchange.
        selector = random.randint(0, len(self.deserialized_interchange_variable_mappings)-1)
        interchange_variable_mapping = self.deserialized_interchange_variable_mappings[selector]
        teacher_variable_names = random.choice(interchange_variable_mapping[0])
        student_variable_names = random.choice(interchange_variable_mapping[1])
        teacher_interchanged_variables_mapping = {}
        student_interchanged_variables_mapping = {}
        # we need to do the interchange here.
        for i, variable in enumerate(teacher_variable_names):
            layer_index, head_index, LOC = parse_variable_name(variable)
            if layer_index in teacher_interchanged_variables_mapping:
                teacher_interchanged_variables_mapping[layer_index] += [(i, head_index, LOC)]
            else:
                teacher_interchanged_variables_mapping[layer_index] = [(i, head_index, LOC)]
        for i, variable in enumerate(student_variable_names):
            layer_index, head_index, LOC = parse_variable_name(variable)
            if layer_index in student_interchanged_variables_mapping:
                student_interchanged_variables_mapping[layer_index] += [(i, head_index, LOC)]
            else:
                student_interchanged_variables_mapping[layer_index] = [(i, head_index, LOC)]
        
        if self.mlm:
            with torch.no_grad():
                # teacher forward pass normal.
                teacher_outputs = self.teacher(
                    input_ids=input_ids, attention_mask=attention_mask
                )  # (bs, seq_length, voc_size)
                # dual on main example
                # teacher forward pass for interchange variables.
                dual_counterfactual_activations_teacher = get_activation_at(
                    self.teacher,
                    dual_input_ids, # this is different!
                    dual_attention_mask, # this is different!
                    variable_names=teacher_variable_names
                )
                # teacher forward pass for interchanged outputs.
                counterfactual_outputs_teacher = self.teacher(
                    input_ids=input_ids, # this is different!
                    attention_mask=attention_mask, # this is different!
                    interchanged_variables=dual_counterfactual_activations_teacher,
                    variable_names=teacher_interchanged_variables_mapping,
                    sampled_interchange_position=sampled_interchange_position,
                )
            t_logits, t_hidden_states = \
                teacher_outputs["logits"], teacher_outputs["hidden_states"]
            student_outputs = self.student(
                input_ids=input_ids, attention_mask=attention_mask,
                t_logits=t_logits,
                t_hidden_states=t_hidden_states,
                temperature=self.temperature,
                restrict_ce_to_mask=self.params.restrict_ce_to_mask,
                lm_labels=lm_labels,
                alpha_mlm=self.alpha_mlm,
                alpha_clm=self.alpha_clm,
                alpha_mse=self.alpha_mse,
                alpha_cos=self.alpha_cos,
            )  # (bs, seq_length, voc_size)
            s_logits, s_hidden_states = student_outputs["logits"], student_outputs["hidden_states"]
            causal_t_logits, causal_t_hidden_states = \
                counterfactual_outputs_teacher["logits"], counterfactual_outputs_teacher["hidden_states"]
        else:
            assert False # we are not supporting this branch!
        
        # standard losses.
        loss_ce = student_outputs["loss_ce"].mean() if self.multi_gpu else student_outputs["loss_ce"]
        loss = self.alpha_ce * loss_ce

        if self.alpha_mlm > 0.0:
            loss_mlm = student_outputs["loss_mlm"].mean() if self.multi_gpu else student_outputs["loss_mlm"]
            loss += self.alpha_mlm * loss_mlm
        if self.alpha_clm > 0.0:
            loss_clm = student_outputs["loss_clm"].mean() if self.multi_gpu else student_outputs["loss_clm"]
            loss += self.alpha_clm * loss_clm
        if self.alpha_mse > 0.0:
            loss_mse = student_outputs["loss_mse"].mean() if self.multi_gpu else student_outputs["loss_mse"]
            loss += self.alpha_mse * loss_mse
        if self.alpha_cos > 0.0:
            loss_cos = student_outputs["loss_cos"].mean() if self.multi_gpu else student_outputs["loss_cos"]
            loss += self.alpha_cos * loss_cos
            
        # we need to get causal distillation loss!
        dual_counterfactual_activations_student = get_activation_at(
            self.student,
            dual_input_ids, # this is different!
            dual_attention_mask, # this is different!
            variable_names=student_variable_names
        )
        # dual on main.
        counterfactual_outputs_student = self.student(
            input_ids=input_ids, # this is different!
            attention_mask=attention_mask, # this is different!
            # interchange.
            interchanged_variables=dual_counterfactual_activations_student,
            variable_names=student_interchanged_variables_mapping,
            sampled_interchange_position=sampled_interchange_position,
            # loss.
            t_logits=t_logits,
            t_hidden_states=t_hidden_states,
            causal_t_logits=causal_t_logits,
            causal_t_hidden_states=causal_t_hidden_states,
            s_logits=s_logits,
            s_hidden_states=s_hidden_states,
            temperature=self.temperature,
            restrict_ce_to_mask=self.params.restrict_ce_to_mask,
        )
        # sanity check.
        assert "loss_ce" not in counterfactual_outputs_student
        assert "loss_mlm" not in counterfactual_outputs_student
        assert "loss_clm" not in counterfactual_outputs_student
        assert "loss_mse" not in counterfactual_outputs_student
        assert "loss_cos" not in counterfactual_outputs_student
        causal_loss_ce = counterfactual_outputs_student["causal_loss_ce"].mean() if self.multi_gpu else counterfactual_outputs_student["causal_loss_ce"]
        teacher_interchange_efficacy = \
            counterfactual_outputs_student["teacher_interchange_efficacy"].mean() if self.multi_gpu else counterfactual_outputs_student["teacher_interchange_efficacy"]
        student_interchange_efficacy = \
            counterfactual_outputs_student["student_interchange_efficacy"].mean() if self.multi_gpu else counterfactual_outputs_student["student_interchange_efficacy"]
        if self.alpha_causal > 0.0:
            loss += self.alpha_causal * causal_loss_ce
                
        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_loss_ce = loss_ce.item()
        if self.alpha_mlm > 0.0:
            self.last_loss_mlm = loss_mlm.item()
        if self.alpha_clm > 0.0:
            self.last_loss_clm = loss_clm.item()
        if self.alpha_mse > 0.0:
            self.last_loss_mse = loss_mse.item()
        if self.alpha_cos > 0.0:
            self.last_loss_cos = loss_cos.item()
        # optional recording of the value.
        self.last_loss_causal_ce = causal_loss_ce.item()
        # record efficacy of the interchange.
        self.last_teacher_interchange_efficacy = teacher_interchange_efficacy.item()
        self.last_student_interchange_efficacy = student_interchange_efficacy.item()
            
        self.optimize(loss, skip_update_iter=skip_update_iter)

        self.n_sequences_epoch += input_ids.size(0)
        
    def _step_parallel(
        self, input_ids: torch.tensor, 
        attention_mask: torch.tensor, 
        lm_labels: torch.tensor,
        dual_input_ids: torch.tensor, 
        dual_attention_mask: torch.tensor, 
        dual_lm_labels: torch.tensor,
        sampled_interchange_position: torch.tensor,
    ):
        """
        WARNING: Parallel requires GPUs with larger memory. It involves computations across two examples
        with two parallel iterations.
        
        One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).

        Input:
        ------
        input_ids/dual_input_ids: `torch.tensor(bs, seq_length)` - The token ids.
        attention_mask/dual_attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
        lm_labels/dual_lm_labels: `torch.tensor(bs, seq_length)` - The language modeling labels (mlm labels for MLM and clm labels for CLM).
        """
        
        # preparing for causal distillation.
        # we randomly select the pool of neurons to interchange.
        selector = random.randint(0, len(self.deserialized_interchange_variable_mappings)-1)
        interchange_variable_mapping = self.deserialized_interchange_variable_mappings[selector]
        teacher_variable_names = random.choice(interchange_variable_mapping[0])
        student_variable_names = random.choice(interchange_variable_mapping[1])
        teacher_interchanged_variables_mapping = {}
        student_interchanged_variables_mapping = {}
        # we need to do the interchange here.
        for i, variable in enumerate(teacher_variable_names):
            layer_index, head_index, LOC = parse_variable_name(variable)
            if layer_index in teacher_interchanged_variables_mapping:
                teacher_interchanged_variables_mapping[layer_index] += [(i, head_index, LOC)]
            else:
                teacher_interchanged_variables_mapping[layer_index] = [(i, head_index, LOC)]
        for i, variable in enumerate(student_variable_names):
            layer_index, head_index, LOC = parse_variable_name(variable)
            if layer_index in student_interchanged_variables_mapping:
                student_interchanged_variables_mapping[layer_index] += [(i, head_index, LOC)]
            else:
                student_interchanged_variables_mapping[layer_index] = [(i, head_index, LOC)]
        
        if self.mlm:
            with torch.no_grad():
                # teacher forward pass normal.
                teacher_outputs = self.teacher(
                    input_ids=input_ids, attention_mask=attention_mask
                )  # (bs, seq_length, voc_size)
                # teacher forward pass normal for the dual example.
                dual_teacher_outputs = self.teacher(
                    input_ids=dual_input_ids, attention_mask=dual_attention_mask
                )  # (bs, seq_length, voc_size)
                
                # dual on main example
                # teacher forward pass for interchange variables.
                dual_counterfactual_activations_teacher = get_activation_at(
                    self.teacher,
                    dual_input_ids, # this is different!
                    dual_attention_mask, # this is different!
                    variable_names=teacher_variable_names
                )
                # teacher forward pass for interchanged outputs.
                counterfactual_outputs_teacher = self.teacher(
                    input_ids=input_ids, # this is different!
                    attention_mask=attention_mask, # this is different!
                    interchanged_variables=dual_counterfactual_activations_teacher,
                    variable_names=teacher_interchanged_variables_mapping,
                    sampled_interchange_position=sampled_interchange_position,
                )   
                
                # main on dual example
                # teacher forward pass for interchange variables.
                counterfactual_activations_teacher = get_activation_at(
                    self.teacher,
                    input_ids, # this is different!
                    attention_mask, # this is different!
                    variable_names=teacher_variable_names
                )
                # teacher forward pass for interchanged outputs.
                dual_counterfactual_outputs_teacher = self.teacher(
                    input_ids=dual_input_ids, # this is different!
                    attention_mask=dual_attention_mask, # this is different!
                    interchanged_variables=counterfactual_activations_teacher,
                    variable_names=teacher_interchanged_variables_mapping,
                    sampled_interchange_position=sampled_interchange_position,
                )
            t_logits, t_hidden_states = \
                teacher_outputs["logits"], teacher_outputs["hidden_states"]
            dual_t_logits, dual_t_hidden_states = \
                dual_teacher_outputs["logits"], dual_teacher_outputs["hidden_states"]
            student_outputs = self.student(
                input_ids=input_ids, attention_mask=attention_mask,
                t_logits=t_logits,
                t_hidden_states=t_hidden_states,
                temperature=self.temperature,
                restrict_ce_to_mask=self.params.restrict_ce_to_mask,
                lm_labels=lm_labels,
                alpha_mlm=self.alpha_mlm,
                alpha_clm=self.alpha_clm,
                alpha_mse=self.alpha_mse,
                alpha_cos=self.alpha_cos,
            )  # (bs, seq_length, voc_size)
            dual_student_outputs = self.student(
                input_ids=dual_input_ids, attention_mask=dual_attention_mask,
                t_logits=dual_t_logits,
                t_hidden_states=dual_t_hidden_states,
                temperature=self.temperature,
                restrict_ce_to_mask=self.params.restrict_ce_to_mask,
                lm_labels=lm_labels,
                alpha_mlm=self.alpha_mlm,
                alpha_clm=self.alpha_clm,
                alpha_mse=self.alpha_mse,
                alpha_cos=self.alpha_cos,
            )  # (bs, seq_length, voc_size)
            s_logits, s_hidden_states = student_outputs["logits"], student_outputs["hidden_states"]
            dual_s_logits, dual_s_hidden_states = student_outputs["logits"], student_outputs["hidden_states"]
            causal_t_logits, causal_t_hidden_states = \
                counterfactual_outputs_teacher["logits"], counterfactual_outputs_teacher["hidden_states"]
            dual_causal_t_logits, dual_causal_t_hidden_states = \
                counterfactual_outputs_teacher["logits"], counterfactual_outputs_teacher["hidden_states"]
        else:
            assert False # we are not supporting this branch!
        
        # standard losses.
        loss_ce = student_outputs["loss_ce"].mean() if self.multi_gpu else student_outputs["loss_ce"]
        loss_ce += dual_student_outputs["loss_ce"].mean() if self.multi_gpu else dual_student_outputs["loss_ce"]
        loss = self.alpha_ce * loss_ce

        if self.alpha_mlm > 0.0:
            loss_mlm = student_outputs["loss_mlm"].mean() if self.multi_gpu else student_outputs["loss_mlm"]
            loss_mlm += dual_student_outputs["loss_mlm"].mean() if self.multi_gpu else dual_student_outputs["loss_mlm"]
            loss += self.alpha_mlm * loss_mlm
        if self.alpha_clm > 0.0:
            loss_clm = student_outputs["loss_clm"].mean() if self.multi_gpu else student_outputs["loss_clm"]
            loss_clm += dual_student_outputs["loss_clm"].mean() if self.multi_gpu else dual_student_outputs["loss_clm"]
            loss += self.alpha_clm * loss_clm
        if self.alpha_mse > 0.0:
            loss_mse = student_outputs["loss_mse"].mean() if self.multi_gpu else student_outputs["loss_mse"]
            loss_mse += dual_student_outputs["loss_mse"].mean() if self.multi_gpu else dual_student_outputs["loss_mse"]
            loss += self.alpha_mse * loss_mse
        if self.alpha_cos > 0.0:
            loss_cos = student_outputs["loss_cos"].mean() if self.multi_gpu else student_outputs["loss_cos"]
            loss_cos += dual_student_outputs["loss_cos"].mean() if self.multi_gpu else dual_student_outputs["loss_cos"]
            loss += self.alpha_cos * loss_cos
            
       # we need to get causal distillation loss!
        dual_counterfactual_activations_student = get_activation_at(
            self.student,
            dual_input_ids, # this is different!
            dual_attention_mask, # this is different!
            variable_names=student_variable_names
        )
        counterfactual_activations_student = get_activation_at(
            self.student,
            input_ids, # this is different!
            attention_mask, # this is different!
            variable_names=student_variable_names
        )
        # dual on main.
        counterfactual_outputs_student = self.student(
            input_ids=input_ids, # this is different!
            attention_mask=attention_mask, # this is different!
            # interchange.
            interchanged_variables=dual_counterfactual_activations_student,
            variable_names=student_interchanged_variables_mapping,
            sampled_interchange_position=sampled_interchange_position,
            # loss.
            t_logits=t_logits,
            t_hidden_states=t_hidden_states,
            causal_t_logits=causal_t_logits,
            causal_t_hidden_states=causal_t_hidden_states,
            s_logits=s_logits,
            s_hidden_states=s_hidden_states,
            temperature=self.temperature,
            restrict_ce_to_mask=self.params.restrict_ce_to_mask,
        )
        # main on dual.
        dual_counterfactual_outputs_student = self.student(
            input_ids=dual_input_ids, # this is different!
            attention_mask=dual_attention_mask, # this is different!
            # interchange.
            interchanged_variables=dual_counterfactual_activations_student,
            variable_names=student_interchanged_variables_mapping,
            sampled_interchange_position=sampled_interchange_position,
            # loss.
            t_logits=dual_t_logits,
            t_hidden_states=dual_t_hidden_states,
            causal_t_logits=dual_causal_t_logits,
            causal_t_hidden_states=dual_causal_t_hidden_states,
            s_logits=dual_s_logits,
            s_hidden_states=dual_s_hidden_states,
            temperature=self.temperature,
            restrict_ce_to_mask=self.params.restrict_ce_to_mask,
        )
        
        # sanity check.
        assert "loss_ce" not in counterfactual_outputs_student and "loss_ce" not in dual_counterfactual_outputs_student
        assert "loss_mlm" not in counterfactual_outputs_student and "loss_mlm" not in dual_counterfactual_outputs_student
        assert "loss_clm" not in counterfactual_outputs_student and "loss_clm" not in dual_counterfactual_outputs_student
        assert "loss_mse" not in counterfactual_outputs_student and "loss_mse" not in dual_counterfactual_outputs_student
        assert "loss_cos" not in counterfactual_outputs_student and "loss_cos" not in dual_counterfactual_outputs_student
        causal_loss_ce = counterfactual_outputs_student["causal_loss_ce"].mean() if self.multi_gpu else counterfactual_outputs_student["causal_loss_ce"]
        causal_loss_ce += dual_counterfactual_outputs_student["causal_loss_ce"].mean() if self.multi_gpu else dual_counterfactual_outputs_student["causal_loss_ce"]
        teacher_interchange_efficacy = \
            counterfactual_outputs_student["teacher_interchange_efficacy"].mean() if self.multi_gpu else counterfactual_outputs_student["teacher_interchange_efficacy"]
        student_interchange_efficacy = \
            counterfactual_outputs_student["student_interchange_efficacy"].mean() if self.multi_gpu else counterfactual_outputs_student["student_interchange_efficacy"]
        teacher_interchange_efficacy += \
            dual_counterfactual_outputs_student["teacher_interchange_efficacy"].mean() if self.multi_gpu else dual_counterfactual_outputs_student["teacher_interchange_efficacy"]
        student_interchange_efficacy += \
            dual_counterfactual_outputs_student["student_interchange_efficacy"].mean() if self.multi_gpu else dual_counterfactual_outputs_student["student_interchange_efficacy"]
        if self.alpha_causal > 0.0:
            loss += self.alpha_causal * causal_loss_ce
                
        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_loss_ce = loss_ce.item()
        if self.alpha_mlm > 0.0:
            self.last_loss_mlm = loss_mlm.item()
        if self.alpha_clm > 0.0:
            self.last_loss_clm = loss_clm.item()
        if self.alpha_mse > 0.0:
            self.last_loss_mse = loss_mse.item()
        if self.alpha_cos > 0.0:
            self.last_loss_cos = loss_cos.item()
        # optional recording of the value.
        self.last_loss_causal_ce = causal_loss_ce.item()
        # record efficacy of the interchange.
        self.last_teacher_interchange_efficacy = teacher_interchange_efficacy.item()
        self.last_student_interchange_efficacy = student_interchange_efficacy.item()
            
        self.optimize(loss)

        self.n_sequences_epoch += input_ids.size(0)

    def optimize(self, loss, skip_update_iter=False):
        """
        Normalization on the loss (gradient accumulation or distributed training), followed by
        backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
        Also update the metrics for tensorboard.
        """
        # Check for NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        if self.multi_gpu:
            loss = loss.mean()
        if self.params.gradient_accumulation_steps > 1:
            loss = loss / self.params.gradient_accumulation_steps

        if self.fp16:
            from apex import amp

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        """
        In case where we want to do two mini-steps for dual on main interchange,
        and main on dual interchange (including normal objectives), we want to
        skip the iter update, so the gradients are accumulated within the step
        which includes gradients from two mini-steps.
        """
        self.iter(skip_update_iter=skip_update_iter)

        if self.n_iter % self.params.gradient_accumulation_steps == 0:
            if self.fp16:
                nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.params.max_grad_norm)
            else:
                nn.utils.clip_grad_norm_(self.student.parameters(), self.params.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

    def iter(self, skip_update_iter=False):
        """
        Update global counts, write to tensorboard and save checkpoint.
        """
        
        if not skip_update_iter:
            self.n_iter += 1
            self.n_total_iter += 1
            if self.n_total_iter % self.params.checkpoint_interval == 0:
                self.save_checkpoint()
        
        """
        Logging is not affected by the flag skip_update_iter.
        We want to log crossway effects, and losses should be
        in the same magnitude.
        """
        if self.n_total_iter % self.params.log_interval == 0:
            self.log_tensorboard()
            self.last_log = time.time()

    def log_tensorboard(self):
        """
        Log into tensorboard. Only by the master process.
        """
        if not self.is_master:
            return
        
        if not self.is_wandb:
            return

        wandb.log(
            {
                "train/cum_avg_loss_epoch": self.total_loss_epoch / self.n_iter, 
                "train/loss": self.last_loss, 
                "train/loss_ce": self.last_loss_ce, 
            }, 
            step=self.n_total_iter
        )
        
        if self.alpha_mlm > 0.0:
            wandb.log(
                {"train/loss_mlm": self.last_loss_mlm}, 
                step=self.n_total_iter
            )
        if self.alpha_clm > 0.0:
            wandb.log(
                {"train/loss_clm": self.last_loss_clm}, 
                step=self.n_total_iter
            )
        if self.alpha_mse > 0.0:
            wandb.log(
                {"train/loss_mse": self.last_loss_mse}, 
                step=self.n_total_iter
            )
        if self.alpha_cos > 0.0:
            wandb.log(
                {"train/loss_cos": self.last_loss_cos}, 
                step=self.n_total_iter
            )

        wandb.log(
            {
                "train/loss_causal_ce": self.last_loss_causal_ce,
                "train/teacher_interchange_efficacy": self.last_teacher_interchange_efficacy,
                "train/student_interchange_efficacy": self.last_student_interchange_efficacy,
            }, 
            step=self.n_total_iter
        )
        
        wandb.log(
            {
                "train/learning_rate": self.scheduler.get_lr()[0],
                "train/memory_usage": psutil.virtual_memory()._asdict()["used"] / 1_000_000,
                "train/speed": time.time() - self.last_log,
            }, 
            step=self.n_total_iter
        )

    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """
        logger.info(f"{self.n_sequences_epoch} sequences have been trained during this epoch.")

        if self.is_master:
            self.save_checkpoint(checkpoint_name=f"model_epoch_{self.epoch}.pth")
            if self.is_wandb:
                wandb.log(
                    {
                        "epoch/loss": self.total_loss_epoch / self.n_iter, 
                        'epoch': self.epoch
                    }
                )

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.total_loss_epoch = 0

    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """
        Save the current state. Only by the master process.
        """
        if not self.is_master:
            return
        mdl_to_save = self.student.module if hasattr(self.student, "module") else self.student
        mdl_to_save.config.save_pretrained(self.dump_path)
        state_dict = mdl_to_save.state_dict()
        torch.save(state_dict, os.path.join(self.dump_path, checkpoint_name))