![Python 3.7](https://img.shields.io/badge/python-3.7-blueviolet.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-MIT-05b502.svg?style=plastic)

# Causal Distillation for Language Models (DIITO)
<p align="center">
  <b><a href="https://zen-wu.social/">Zhengxuan Wu</a>*,<a href="https://atticusg.github.io/">Atticus Geiger</a>*, <a href="https://www.linkedin.com/in/jsrozner/">Josh Rozner</a>, <a href="https://www.elisakreiss.com/">Elisa Kreiss</a>, <a href="https://www.linkedin.com/in/hansonhxlu/">Hanson Lu</a>, <a href="https://web.stanford.edu/~icard/">Thomas Icard</a>, <a href="https://web.stanford.edu/~cgpotts/">Christopher Potts</a>, <a href="https://cocolab.stanford.edu/ndg">Noah D. Goodman</a></b></span>
</p>

<div align="center">
  <img src="https://i.ibb.co/Q8NNHPJ/Screen-Shot-2021-12-06-at-4-53-28-PM.png" style="float:left" width="800px">
</div>
<p></p>

The is an implementation of our preprint [Causal Distillation for Language Models](https://zen-wu.social/papers/ACL22_CausalDistill.pdf). The standard approach to distillation trains a student model against two objectives: a task-specific objective (e.g., language modeling) and an imitation objective that encourages the hidden states of the student model to be similar to those of the larger teacher model. In this paper, we show that it is beneficial to augment distillation with a third objective that encourages the student to imitate the causal computation process of the teacher through interchange intervention training (IIT). We name our method **the distillation interchange intervention training objective (DIITO)**.

**We find DIITO is helpful in a low-resource setting. DIITO performs on-par with (97%) standard distillation but training with 97% less of data.**

We fork our main codebase from the [Huggingface Distillation Interface](https://github.com/huggingface/transformers/tree/master/examples/research_projects/distillation).

## Release Notes
:white_check_mark: 12/02/2021 Our paper on [Interchange Intervention Training (IIT)](https://arxiv.org/abs/2112.00826) is released! Read this for a more formal definition of the method.   
:white_check_mark: 12/06/2021 Released the causal distillation codebase with [the preprint](https://arxiv.org/abs/2112.02505).   
:white_check_mark: 12/06/2021 Released evaluation results on distilled tiny-BERT (3 layers) with the Wiki-Text 103M dataset.  
:white_check_mark: 01/14/2022 Released newer version of **DIITO**, and its evaluation results. You can view our privately shared [updated preprint](https://zen-wu.social/papers/ACL22_CausalDistill.pdf) for more details. 
:white_check_mark: 02/21/2022 Released the codebase for [**DIITO-XXS**](https://github.com/frankaging/Causal-Distill-XXS) that applies DITTO to distill task-specific models in NLP with a focus on supporting model distillation in a low-resource setting. Check out the repo for more info!
⬜️ Released DIITO (6 layers) model trained with English Wikipedia + Bookcorpus.   

If you experience any issues or have suggestions, please contact me either thourgh the issues page or at wuzhengx@stanford.edu. 

## Benchmark Results
Here are the results on the dev sets of GLUE:

| Model                     |# of Training Tokens| Average-score                  | CoLA | MNLI | MRPC | QNLI | QQP  | RTE  | SST-2| STS-B |
| :---:                     | :---: |    :---:                       | :---:| :---:| :---:| :---:| :---:| :---:| :---:| :---:|
| DistilBERT (6 layers) [Devlin et al., 2019](https://arxiv.org/abs/1910.01108)     | 3.3B  |  **79.59**          | 51.30 | 82.10 | 87.50 | 89.20 | 88.50 | 59.90 | 91.30 | 86.90 |  
| DistilBERT (6 layers)     | 0.1B  |  **75.80**          | 40.43 | 78.95 | 87.45 | 84.76 | 84.96 | 60.10 | 89.38 | 80.40 |
| DIITO (6 layers)     | 0.1B  |  **77.14**          |  45.17 | 79.68 | 88.18 | 85.83 | 85.31 | 60.94 | 90.32 | 81.69 |
| DIITO (6 layers)     | 3.3B  |      (-)      | (-) | (-)| (-) | (-) | (-) | (-) | (-) | (-) |

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
        eprint={2112.02505},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
  }

```

## Requirements
- Python 3.6 or 3.7 are supported.
- Pytorch Version: 1.9.0
- Transfermers Version: 4.11.3
- Datasets Version: Version: 1.8.0
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
CUDA_VISIBLE_DEVICES=0,1,2,3 python causal_train.py \
--force \
--n_gpu 4 \
--log_interval 10 \
--student_type distilbert \
--student_config ./training_configs/distilbert-base-uncased-large.json \
--student_pretrained_weights ./distillation_checkpoints/bert-base-uncased_num_layer_6.pth \
--teacher_type bert \
--teacher_name bert-base-uncased \
--neuron_mapping ./training_configs/single_middle_layer_6.nm \
--mlm --alpha_ce 0.25 --alpha_mlm 0.25 --alpha_cos 0.25 --alpha_clm 0.0 --alpha_causal_ce 0.25 --alpha_causal_cos 0.0 \
--interchange_prop 0.3 --interchange_max_token -1 --interchange_consecutive_only \
--freeze_pos_embs \
--dump_path ./results/ \
--data_file ./wikitext-dataset/binarized_text.train.bert-base-uncased.pickle \
--token_counts ./wikitext-dataset/binarized_text.train.token_counts.bert-base-uncased.pickle \
--seed 42 \
--n_epoch 3 \
--gradient_accumulation_steps 6 \
--batch_size 40
```
Note that you can simply turn our causal distillation objective on/off through setting the arguments. For instance, we recently add this argument `--alpha_causal_cos` to support causal loss on the cosine loss term. Note that the effective batch size in our setting is set to 240.

## Evaluation
After you get your distilled models, you need to fine-tune them and evaluate them with downstream tasks. We provide you all the scripts you need to run.

### MLM Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python run_mlm.py \
--model_name_or_path ./path_to_your_model/ \
--dataset_dir ../path_to_your_data/ \
--tokenizer_name bert-base-uncased \
--do_eval \
--output_dir /tmp/test-mlm \
--cache_dir ./distill_cache/
```

### GLUE Evaluation
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_glue.py \
--model_name_or_path ./path_to_your_model/ \
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
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_ner.py \
--model_name_or_path ./path_to_your_model/ \
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
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_qa.py \
--model_name_or_path ./path_to_your_model/ \
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
