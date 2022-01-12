import itertools
import os

seeds = [111,222,333,444,555]
model_dirs = [
"s_distilbert_t_bert_data_wikitext-dataset_seed_42_mlm_True_ce_0.25_mlm_0.25_cos_0.25_causal-ce_0.25_causal-cos_0.0_nm_multiple_single_middle_late_layer_6_crossway_False_int-prop_0.3_consec-token_True_masked-token_False_max-int-token_-1_eff-bs_240",
"s_distilbert_t_bert_data_wikitext-dataset_seed_42_mlm_True_ce_0.25_mlm_0.25_cos_0.25_causal-ce_0.25_causal-cos_0.0_nm_multiple_single_multilayer_layer_6_crossway_False_int-prop_0.3_consec-token_True_masked-token_False_max-int-token_-1_eff-bs_240",
"s_distilbert_t_bert_data_wikitext-dataset_seed_42_mlm_True_ce_0.25_mlm_0.25_cos_0.25_causal-ce_0.25_causal-cos_0.0_nm_single_middle_layer_6_crossway_False_int-prop_0.3_consec-token_True_masked-token_False_max-int-token_-1_eff-bs_240",
"s_distilbert_t_bert_data_wikitext-dataset_seed_42_mlm_True_ce_0.33_mlm_0.33_cos_0.33_causal-ce_0.0_causal-cos_0.0_nm_single_middle_layer_6_crossway_False_int-prop_0.3_consec-token_True_masked-token_False_max-int-token_-1_eff-bs_240",
"s_distilbert_t_bert_data_wikitext-dataset_seed_66_mlm_True_ce_0.25_mlm_0.25_cos_0.25_causal-ce_0.25_causal-cos_0.0_nm_multiple_single_middle_late_layer_6_crossway_False_int-prop_0.3_consec-token_True_masked-token_False_max-int-token_-1_eff-bs_240",
"s_distilbert_t_bert_data_wikitext-dataset_seed_66_mlm_True_ce_0.25_mlm_0.25_cos_0.25_causal-ce_0.25_causal-cos_0.0_nm_multiple_single_multilayer_layer_6_crossway_False_int-prop_0.3_consec-token_True_masked-token_False_max-int-token_-1_eff-bs_240",
"s_distilbert_t_bert_data_wikitext-dataset_seed_66_mlm_True_ce_0.25_mlm_0.25_cos_0.25_causal-ce_0.25_causal-cos_0.0_nm_single_middle_layer_6_crossway_False_int-prop_0.3_consec-token_True_masked-token_False_max-int-token_-1_eff-bs_252",
"s_distilbert_t_bert_data_wikitext-dataset_seed_66_mlm_True_ce_0.33_mlm_0.33_cos_0.33_causal-ce_0.0_causal-cos_0.0_nm_single_middle_layer_6_crossway_False_int-prop_0.3_consec-token_True_masked-token_False_max-int-token_-1_eff-bs_240",
"s_distilbert_t_bert_data_wikitext-dataset_seed_88_mlm_True_ce_0.25_mlm_0.25_cos_0.25_causal-ce_0.25_causal-cos_0.0_nm_multiple_single_middle_late_layer_6_crossway_False_int-prop_0.3_consec-token_True_masked-token_False_max-int-token_-1_eff-bs_240",
"s_distilbert_t_bert_data_wikitext-dataset_seed_88_mlm_True_ce_0.25_mlm_0.25_cos_0.25_causal-ce_0.25_causal-cos_0.0_nm_multiple_single_multilayer_layer_6_crossway_False_int-prop_0.3_consec-token_True_masked-token_False_max-int-token_-1_eff-bs_240",
"s_distilbert_t_bert_data_wikitext-dataset_seed_88_mlm_True_ce_0.25_mlm_0.25_cos_0.25_causal-ce_0.25_causal-cos_0.0_nm_single_middle_layer_6_crossway_False_int-prop_0.3_consec-token_True_masked-token_False_max-int-token_-1_eff-bs_240",
"s_distilbert_t_bert_data_wikitext-dataset_seed_88_mlm_True_ce_0.33_mlm_0.33_cos_0.33_causal-ce_0.0_causal-cos_0.0_nm_single_middle_layer_6_crossway_False_int-prop_0.3_consec-token_True_masked-token_False_max-int-token_-1_eff-bs_240",
]
for i in range(len(seeds)):
    for j in range(len(model_dirs)):
        
        command = f'CUDA_VISIBLE_DEVICES=0,4,6,9 python run_qa.py \
                    --model_name_or_path ./post_arxiv_mim_results/{model_dirs[j]}/ \
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
                    --output_dir ./qa_post_arxiv_mim_results/ \
                    --report_to none \
                    --seed {seeds[i]}'
        print(command)
        os.system(command)