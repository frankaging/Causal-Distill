from glob import glob
import os

eval_model_path = "glue_post_arxiv_mim_results"
eval_method = "do_eval"

for path in glob(f"{eval_model_path}/*/"):
    if "ce_0.25_mlm_0.25_cos_0.25_causal-ce_0.25_causal-cos_0.0_nm_single_middle_layer_6_crossway_False" in path:
        print(f"generating results for path at: {path}")
        cmd = f"CUDA_VISIBLE_DEVICES=4,5,7,8 python run_glue.py \
              --model_name_or_path {path} --tokenizer_name bert-base-uncased \
              --{eval_method} --per_device_eval_batch_size 32 --max_seq_length 128 \
              --output_dir ../eval_glue_results/ --cache_dir ./distill_cache/ \
              --report_to none"
        print(f"starting command: {cmd}")
        os.system(cmd)