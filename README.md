![Python 3.7](https://img.shields.io/badge/python-3.7-blueviolet.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-MIT-05b502.svg?style=plastic)

# Causal Distillation for Language Models
<p align="center">
  <b><a href="https://zen-wu.social/">Zhengxuan Wu</a>*,<a href="https://atticusg.github.io/">Atticus Geiger</a>*, <a href="https://www.linkedin.com/in/jsrozner/">Josh Rozner</a>, <a href="https://www.elisakreiss.com/">Elisa Kreiss</a>, <a href="https://www.linkedin.com/in/hansonhxlu/">Hanson Lu</a>, <a href="https://web.stanford.edu/~icard/">Thomas Icard</a>, <a href="https://web.stanford.edu/~cgpotts/">Christopher Potts</a>, <a href="https://cocolab.stanford.edu/ndg">Noah D. Goodman</a></b></span>
</p>

<div align="center">
  <img src="https://i.ibb.co/Q8NNHPJ/Screen-Shot-2021-12-06-at-4-53-28-PM.png" style="float:left" width="800px">
</div>
<p></p>

The is an implementation of our preprint [Causal Distillation for Language Models](https://zen-wu.social/papers/ACL22_CausalDistill.pdf). The standard approach to distillation trains a student model against two objectives: a task-specific objective (e.g., language modeling) and an imitation objective that encourages the hidden states of the student model to be similar to those of the larger teacher model. In this paper, we show that it is beneficial to augment distillation with a third objective that encourages the student to imitate the causal computation process of the teacher through **interchange intervention training (IIT)**. 

We fork our main codebase from the [Huggingface Distillation Interface](https://github.com/huggingface/transformers/tree/master/examples/research_projects/distillation).

## Release Notes
:white_check_mark: 12/02/2021 Our paper on [Interchange Intervention Training (IIT)](https://arxiv.org/abs/2112.00826) is released! Read this more formal definition of the method.   
:white_check_mark: 12/06/2021 Released the causal distillation codebase with [the preprint](https://zen-wu.social/papers/ACL22_CausalDistill.pdf).   
:white_check_mark: 12/06/2021 Released evaluation results on distilled tiny-BERT (3 layers) with the Wiki-Text 103M dataset.   
⬜️ Released evaluation results on causal-distilled tiny-BERT (3 layers) with the Wiki-Text 103M + BookCorpus dataset.   
⬜️ Released evaluation results on causal-distilled BERT (6 layers) with the Wiki-Text 103M + BookCorpus dataset.    
⬜️ Released more ablation studies.   
⬜️ Released causal-distilled tiny-BERT (3 layers) model files.   
⬜️ Released causal-distilled BERT (6 layers) model files.   

If you experience any issues or have suggestions, please contact me either thourgh the issues page or at wuzhengx@stanford.edu. 

## Benchmark Results
Here are the results on the dev sets of GLUE:

| Model                     | Average-score                  | CoLA | MNLI | MRPC | QNLI | QQP  | RTE  | SST-2| STS-B| WNLI              |
| :---:                     |    :---:                       | :---:| :---:| :---:| :---:| :---:| :---:| :---:| :---:| :---:             |
| DistilBERT (3 layers)     |  **67.8**<sup>1</sup>          | 22.8 | 71.6 | 78.2 | 82.1 | 84.3 | 55.4 | 86.5 | 56.7 | 24.2              |
| CausalBERT (3 layers)     |  **69.7**<sup>1</sup>          | 25.0 | 72.9 | 78.6 | 83.1 | 84.9 | 55.4 | 86.9 | 66.5 | 21.5              |

<sup>1</sup> Average-score computed without WNLI.

## Main Contents
* [Citation](#citation)
* [Requirements](#requirements)
* [Dataset](#dataset)
* [Distillation](#distillation)
* [Evaluation](#evaluation)

## Citation
If you use this repository, please cite the following two papers: [paper for interchange intervention training](https://arxiv.org/abs/2112.00826), and [paper for the our distillation method](https://arxiv.org/abs/2109.08994).
```stex
  @article{geiger-etal-2021-iit,
        title={Inducing Causal Structure for Interpretable Neural Networks}, 
        author={Geiger, Atticus and Wu, Zhengxuan and Lu, Hanson and Rozner, Josh and Kreiss, Elisa and Icard, Thomas and Goodman, Noah D. and Potts, Christopher},
        year={2021},
        eprint={2112.00826},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
  }

  @article{wu-etal-2021-distill,
        title={Causal Distillation for Language Models}, 
        author={Wu, Zhengxuan and Geiger, Atticus and Rozner, Josh and Kreiss, Elisa and Lu, Hanson and Icard, Thomas and Potts, Christopher and Goodman, Noah D.},
        year={2021},
        eprint={},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
  }

```

## Requirements
- Python 3.6 or 3.7 are supported.
- Pytorch Version: 1.9.0
- Transfermers Version: 4.11.3
- Datasets Version: Version: 1.8.0
- We have performed experiments on Titan V GPU. We assume 12GB of GPU memory (more memory can expedite training).
- Since we build our codebase off the [Huggingface Distillation Interface](https://github.com/huggingface/transformers/tree/master/examples/research_projects/distillation), please review their doc for requirements.

## Dataset
Following the [Huggingface Distillation Interface](https://github.com/huggingface/transformers/tree/master/examples/research_projects/distillation), we need to pre-process the datasets before we do distillation. You can refer to their repo for details. We adapt their pre-processing scripts, and update with a few improvements. For example, we can now binarize datasets from the Dataset Hub from huggingface directly.

```bash
# preprocessing from disk
python script/binarized_data.py \
--file_path ../../bert-mid-tuning/data-files/wikitext-15M \
--split train \
--field_name text \
--max_parsing_example 1000 \
--tokenizer_type bert \
--tokenizer_name bert-base-uncased \
--dump_file ./data/binarized_text

# preprocessing from huggingface.
python scripts/binarized_data.py \
--dataset_name bookcorpus \
--split train \
--field_name text \
--tokenizer_type bert \
--tokenizer_name bert-base-uncased \
--dump_file bookcorpus-dataset/binarized_text \
--cache_dir ./distill_cache/

python scripts/binarized_data.py \
--dataset_name wikitext \
--split train \
--field_name text \
--tokenizer_type bert \
--tokenizer_name bert-base-uncased \
--dump_file wikitext-dataset/binarized_text \
--cache_dir ./distill_cache/

python scripts/binarized_data.py \
--dataset_name wikitext+bookcorpus \
--split train \
--field_name text \
--tokenizer_type bert \
--tokenizer_name bert-base-uncased \
--dump_file wikitext+bookcorpus-dataset/binarized_text \
--cache_dir ./distill_cache/

# helper scripts to combine two binarized data files
python scripts/data_combinator.py \
--file_path_left ./bookcorpus-dataset/binarized_text.train.bert-base-uncased.pickle \
--file_path_right ./wikitext-dataset/binarized_text.train.bert-base-uncased.pickle \
--split train \
--tokenizer_name bert-base-uncased \
--dump_file wikitext+bookcorpus-dataset/binarized_text

# multiprocessing preprocessor.
python scripts/binarized_data.py \
--dataset_name bookcorpus \
--split train \
--field_name text \
--tokenizer_type bert \
--tokenizer_name bert-base-uncased \
--dump_file bookcorpus-dataset/binarized_text \
--cache_dir ./distill_cache/ \
--fast_process \
--preprocessing_num_workers 48
```

After you get the datasets ready, you need to generate token counts as well.
```bash
python scripts/token_counts.py \
--data_file data/binarized_text.train.bert-base-uncased.pickle \
--token_counts_dump data/binarized_text.train.token_counts.bert-base-uncased.pickle \
--vocab_size 30522
```

## Distillation
Before training, we recommand you to initialize your student model with weights extracted from the teacher model.
```bash
python scripts/extract_distilbert.py \
--model_type bert \
--model_name bert-base-uncased \
--dump_checkpoint ./distillation_checkpoints/bert-base-uncased_num_layer_3.pth \
--num_layers 3
```

Now, here is an example for you to distill with our causal distillation objective or without,
```bash
CUDA_VISIBLE_DEVICES=9,4 python causal_train.py \
--force \
--n_gpu 2 \
--is_wandb \
--log_interval 10 \
--student_type distilbert \
--student_config ./training_configs/distilbert-base-uncased-small.json \
--student_pretrained_weights ./distillation_checkpoints/bert-base-uncased_num_layer_3.pth \
--teacher_type bert \
--teacher_name bert-base-uncased \
--neuron_mapping ./training_configs/single_middle.nm \
--mlm --alpha_ce 0.25 --alpha_mlm 0.25 --alpha_cos 0.25 --alpha_clm 0.0 --alpha_causal 0.25 \
--freeze_pos_embs \
--dump_path ./results/ \
--data_file ./wikitext-15M/binarized_text.train.bert-base-uncased.pickle \
--token_counts ./wikitext-15M/binarized_text.train.token_counts.bert-base-uncased.pickle \
--seed 42 \
--gradient_accumulation_steps 50 \
--n_epoch 3 \
--batch_size 5

CUDA_VISIBLE_DEVICES=0,1,2,3 python causal_train.py \
--force \
--n_gpu 4 \
--is_wandb \
--log_interval 10 \
--student_type distilbert \
--student_config ./training_configs/distilbert-base-uncased-small.json \
--student_pretrained_weights ./distillation_checkpoints/bert-base-uncased_num_layer_3.pth \
--teacher_type bert \
--teacher_name bert-base-uncased \
--neuron_mapping ./training_configs/single_middle.nm \
--mlm --alpha_ce 0.33 --alpha_mlm 0.33 --alpha_cos 0.33 --alpha_clm 0.0 --alpha_causal 0.00 \
--freeze_pos_embs \
--dump_path ./results/ \
--data_file ./wikitext-15M/binarized_text.train.bert-base-uncased.pickle \
--token_counts ./wikitext-15M/binarized_text.train.token_counts.bert-base-uncased.pickle \
--seed 42 \
--gradient_accumulation_steps 124 \
--n_epoch 6 \
--batch_size 4
```
Note that you can simply turn our causal distillation objective on/off through setting the arguments.

## Evaluation
After you get your distilled models, you need to fine-tune them and evaluate them with downstream tasks. We provide you all the scripts you need to run.

### MLM Evaluation
```bash
CUDA_VISIBLE_DEVICES=5 python run_mlm.py \
--model_name_or_path ./results/s_distilbert_t_bert_data_wikitext-15M_seed_42_mlm_True_ce_0.25_mlm_0.25_cos_0.25_causal_0.25_nm_single_multilayer/ \
--dataset_dir ../../bert-mid-tuning/data-files/wikitext-15M/ \
--tokenizer_name bert-base-uncased \
--do_eval \
--output_dir /tmp/test-mlm \
--cache_dir ./distill_cache/
```

### GLUE Evaluation
```bash
CUDA_VISIBLE_DEVICES=5,7,8,9 python run_glue.py \
--model_name_or_path ./results/s_distilbert_t_bert_data_wikitext-dataset_seed_42_mlm_True_ce_0.33_mlm_0.33_cos_0.33_causal_0.0_nm_single_middle/ \
--tokenizer_name bert-base-uncased \
--task_name sst2 \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 32 \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--output_dir ./results/ \
--save_total_limit 1 \
--cache_dir ./distill_cache/
```

### CoNLL Evaluation
```bash
CUDA_VISIBLE_DEVICES=2,3,7,8 python run_ner.py \
--model_name_or_path ./results/s_distilbert_t_bert_data_wikitext-dataset_seed_42_mlm_True_ce_0.33_mlm_0.33_cos_0.33_causal_0.0_nm_single_middle_crossway_False/ \
--tokenizer_name bert-base-uncased \
--dataset_name conll2003 \
--do_train \
--do_eval \
--output_dir ./ner_results/ \
--save_total_limit 1 \
--cache_dir ./distill_cache/
```

### SQuAD Evaluation
```bash
CUDA_VISIBLE_DEVICES=2,3,7,8 python run_qa.py \
--model_name_or_path ./results/s_distilbert_t_bert_data_wikitext-dataset_seed_42_mlm_True_ce_0.33_mlm_0.33_cos_0.33_causal_0.0_nm_single_middle_crossway_False/ \
--tokenizer_name bert-base-uncased \
--dataset_name squad \
--do_train \
--do_eval \
--per_device_train_batch_size 12 \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--max_seq_length 384 \
--doc_stride 128 \
--save_total_limit 1 \
--output_dir ./qa_results/
```
