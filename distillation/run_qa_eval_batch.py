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
        command = f'CUDA_VISIBLE_DEVICES=6,7,8,9 python run_qa.py \
                    --model_name_or_path ./qa_results/squad_{model_dirs[j]}_tseed_{seeds[i]}/ \
                    --tokenizer_name bert-base-uncased \
                    --dataset_name squad \
                    --do_eval \
                    --max_seq_length 384 \
                    --doc_stride 128 \
                    --save_total_limit 1 \
                    --output_dir ./eval_qa_results/'
        print(command)
        os.system(command)