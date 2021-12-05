import itertools
import os

seeds = [111,222,333,444,555]
model_dirs = [
    "s_distilbert_t_bert_data_wikitext-dataset_seed_88_mlm_True_ce_0.25_mlm_0.25_cos_0.25_causal_0.25_nm_multiple_single_middle_late_crossway_False",
    "s_distilbert_t_bert_data_wikitext-dataset_seed_88_mlm_True_ce_0.25_mlm_0.25_cos_0.25_causal_0.25_nm_multiple_single_multilayer_crossway_False",
    "s_distilbert_t_bert_data_wikitext-dataset_seed_88_mlm_True_ce_0.25_mlm_0.25_cos_0.25_causal_0.25_nm_single_middle_crossway_False",
    "s_distilbert_t_bert_data_wikitext-dataset_seed_88_mlm_True_ce_0.33_mlm_0.33_cos_0.33_causal_0.0_nm_single_middle_crossway_False",
    "s_distilbert_t_bert_data_wikitext-dataset_seed_66_mlm_True_ce_0.25_mlm_0.25_cos_0.25_causal_0.25_nm_multiple_single_middle_late_crossway_False",
    "s_distilbert_t_bert_data_wikitext-dataset_seed_66_mlm_True_ce_0.25_mlm_0.25_cos_0.25_causal_0.25_nm_multiple_single_multilayer_crossway_False",
    "s_distilbert_t_bert_data_wikitext-dataset_seed_66_mlm_True_ce_0.25_mlm_0.25_cos_0.25_causal_0.25_nm_single_middle_crossway_False",
    "s_distilbert_t_bert_data_wikitext-dataset_seed_66_mlm_True_ce_0.33_mlm_0.33_cos_0.33_causal_0.0_nm_single_middle_crossway_False",
    "s_distilbert_t_bert_data_wikitext-dataset_seed_42_mlm_True_ce_0.25_mlm_0.25_cos_0.25_causal_0.25_nm_multiple_single_middle_late_crossway_False",
    "s_distilbert_t_bert_data_wikitext-dataset_seed_42_mlm_True_ce_0.25_mlm_0.25_cos_0.25_causal_0.25_nm_multiple_single_multilayer_crossway_False",
    "s_distilbert_t_bert_data_wikitext-dataset_seed_42_mlm_True_ce_0.25_mlm_0.25_cos_0.25_causal_0.25_nm_single_middle_crossway_False",
    "s_distilbert_t_bert_data_wikitext-dataset_seed_42_mlm_True_ce_0.33_mlm_0.33_cos_0.33_causal_0.0_nm_single_middle_crossway_False",
]
for i in range(len(seeds)):
    for j in range(len(model_dirs)):
        command = f'CUDA_VISIBLE_DEVICES=1,2,3,4 python run_ner.py \
                    --model_name_or_path ./results/{model_dirs[j]}/ \
                    --tokenizer_name bert-base-uncased \
                    --dataset_name conll2003 \
                    --do_train \
                    --output_dir ./ner_results/ \
                    --save_total_limit 1 \
                    --cache_dir ./distill_cache/ \
                    --seed {seeds[i]}'
        print(command)
        os.system(command)