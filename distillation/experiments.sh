# system helpers.
nvidia-htop.py


# notebook converter.
jupyter nbconvert --to python causal_train.ipynb
jupyter nbconvert --to python run_mlm.ipynb
jupyter nbconvert --to python run_glue.ipynb


# preprocessing from disk
python scripts/binarized_data.py \
--file_path ../../bert-mid-tuning/data-files/wikitext-15M \
--split train \
--field_name text \
--tokenizer_type bert \
--tokenizer_name bert-base-uncased \
--dump_file wikitext-15M/binarized_text


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


# token count files.
python scripts/token_counts.py \
--data_file wikitext-dataset/binarized_text.train.bert-base-uncased.pickle \
--token_counts_dump wikitext-dataset/binarized_text.train.token_counts.bert-base-uncased.pickle \
--vocab_size 30522

python scripts/token_counts.py \
--data_file bookcorpus-dataset/binarized_text.train.bert-base-uncased.pickle \
--token_counts_dump bookcorpus-dataset/binarized_text.train.token_counts.bert-base-uncased.pickle \
--vocab_size 30522

python scripts/token_counts.py \
--data_file wikitext+bookcorpus-dataset/binarized_text.train.bert-base-uncased.pickle \
--token_counts_dump wikitext+bookcorpus-dataset/binarized_text.train.token_counts.bert-base-uncased.pickle \
--vocab_size 30522


# extract initial weights for the small model.
python scripts/extract_distilbert.py \
--model_type bert \
--model_name bert-base-uncased \
--dump_checkpoint ./distillation_checkpoints/bert-base-uncased_num_layer_3.pth \
--num_layers 3


# demo files.
CUDA_VISIBLE_DEVICES=0,1,3,4 python causal_train.py \
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
--mlm --alpha_ce 0.25 --alpha_mlm 0.25 --alpha_cos 0.25 --alpha_clm 0.0 --alpha_causal 0.25 \
--freeze_pos_embs \
--dump_path ./results/ \
--data_file ./demo_data/binarized_text.train.bert-base-uncased.pickle \
--token_counts ./demo_data/binarized_text.train.token_counts.bert-base-uncased.pickle \
--seed 42 \
--batch_size 8


# distillation pipeline with control condition.
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


# scripts to copy the model file for evaluaiton.
cp model_epoch_0.pth pytorch_model.bin
cp model_epoch_1.pth pytorch_model.bin
cp model_epoch_2.pth pytorch_model.bin
cp checkpoint.pth pytorch_model.bin


# evaluation scripts
CUDA_VISIBLE_DEVICES=5 python run_mlm.py \
--model_name_or_path bert-base-uncased \
--dataset_dir ../../bert-mid-tuning/data-files/wikitext-15M/ \
--tokenizer_name bert-base-uncased \
--do_eval \
--output_dir /tmp/test-mlm \
--cache_dir ./distill_cache/

CUDA_VISIBLE_DEVICES=5 python run_mlm.py \
--model_name_or_path ./results/initial_s/ \
--dataset_dir ../../bert-mid-tuning/data-files/wikitext-15M/ \
--tokenizer_name bert-base-uncased \
--do_eval \
--output_dir /tmp/test-mlm \
--cache_dir ./distill_cache/

CUDA_VISIBLE_DEVICES=5 python run_mlm.py \
--model_name_or_path ./results/s_distilbert_t_bert_data_wikitext-15M_seed_42_mlm_True_ce_0.25_mlm_0.25_cos_0.25_causal_0.25_nm_single_multilayer/ \
--dataset_dir ../../bert-mid-tuning/data-files/wikitext-15M/ \
--tokenizer_name bert-base-uncased \
--do_eval \
--output_dir /tmp/test-mlm \
--cache_dir ./distill_cache/

# run glue task evaluation
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