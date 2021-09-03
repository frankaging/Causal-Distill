# Auto-Distill BERT: Automatically Optimized Ditillation of BERT Model using RL


### Install Requirements
You will have to clone this repo, and install all the dependencies. You can skip this step if you have torch and cuda installed. That is all you need. You can also mannually install these without going through this installation headache that ``requirements.txt`` may give you.
```bash
cd BERT_LRP/code/
pip install -r requirements.txt
```

### Download Pretrained BERT Model
You will have to download pretrained BERT model in order to execute the fine-tune pipeline. We recommand to use models provided by the official release on BERT from [BERT-Base (Google's pre-trained models)](https://github.com/google-research/bert). Note that their model is in tensorflow format. To convert tensorflow model to pytorch model, you can use the helper script to do that. For example,
```bash
cd BERT_LRP/code/
python convert_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path uncased_L-12_H-768_A-12/bert_model.ckpt \
--bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
--pytorch_dump_path uncased_L-12_H-768_A-12/pytorch_model.bin
```
