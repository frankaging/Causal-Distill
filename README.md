# Causal Distillation for Language Models

Codebase for Causal Distillation for Language Models. We fork our main codebase from the [Huggingface Distillation Interface](https://github.com/huggingface/transformers/tree/master/examples/research_projects/distillation). This makes it extremely easy to be intergrated with your customized stuffs.

## Release Notes
* **12/02/2021**: We are preparing to release our code.

## Contents
* [Citation](#citation)
* [Dataset](#dataset)
* [Distillation](#distillation)
* [Evaluation](#evaluation)
* [License](#license)

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

## License

ReaSCAN has a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
