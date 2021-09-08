# Causal Distillation of BERT

### Pre-processing the dataset
```bash
python script/binarized_data.py \
--file_path ../../bert-mid-tuning/data-files/wikitext-15M \
--split train \
--field_name text \
--max_parsing_example 1000 \
--tokenizer_type bert \
--tokenizer_name bert-base-uncased \
--dump_file ./data/binarized_text
```

### Generate token counts
```bash
python scripts/token_counts.py \
--data_file data/binarized_text.train.bert-base-uncased.pickle \
--token_counts_dump data/binarized_text.train.token_counts.bert-base-uncased.pickle \
--vocab_size 30522
```

### Regular Distillation
```bash
```

### Causal Distillation
```bash
```