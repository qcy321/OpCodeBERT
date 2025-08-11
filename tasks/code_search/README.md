# Data Download

## 1.CSN dataset

```bash
cd dataset/CSN
wget https://zenodo.org/record/7857872/files/python.zip
bash run.sh 
cd ../..
```

## 2.AdvTest dataset

```bash
cd dataset && mkdir AdvTest
cd AdvTest
wget https://github.com/microsoft/CodeXGLUE/raw/main/Text-Code/NL-code-search-Adv/dataset.zip
unzip dataset.zip && rm -r dataset.zip && mv dataset/*  && rm -rf dataset
wget https://zenodo.org/record/7857872/files/python.zip
unzip python.zip && python preprocess.py && rm -r python && rm -r *.pkl && rm python.zip
cd ../..
```

# Data preprocessing

Each command only needs to be run once.  
If any issues occur, you can re-download the dataset and run the commands again.

```bash
python data_processing.py --dataset "CSN" --num_processes 8
```

```bash
python data_processing.py --dataset "AdvTest" --num_processes 8
```

```bash
python data_processing.py --dataset "CosQA" --num_processes 8
```

# Fine-Tune

## 1.CSN dataset

```bash
# Training and testing
python run.py \
    --model_name_or_path="XQ112/OpCodeBERT" \
    --dataset="CSN" \
    --output_dir="saved_models/CSN" \
    --do_train \
    --fp16 \
    --do_test \
    --num_train_epochs 10 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --device_ids 0 \
    --seed 42
    
# zero shot
python run.py \
    --model_name_or_path="XQ112/OpCodeBERT" \
    --dataset="CSN" \
    --output_dir="saved_models/CSN" \
    --do_zero \
    --num_train_epochs 10 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --device_ids 0 \
    --seed 42
```

## 2.AdvTest dataset

```bash
# Training and testing
python run.py \
    --model_name_or_path="XQ112/OpCodeBERT" \
    --dataset="AdvTest" \
    --output_dir="saved_models/AdvTest" \
    --do_train \
    --fp16 \
    --do_test \
    --num_train_epochs 2 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --device_ids 0 \
    --seed 42
    
# zero shot
python run.py \
    --model_name_or_path="XQ112/OpCodeBERT" \
    --dataset="AdvTest" \
    --output_dir="saved_models/AdvTest" \
    --do_zero \
    --num_train_epochs 2 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --device_ids 0 \
    --seed 42
```

## 3.CosQA dataset

```bash
# Training and testing
python run.py \
    --model_name_or_path="XQ112/OpCodeBERT" \
    --dataset="CosQA" \
    --output_dir="saved_models/CosQA" \
    --do_train \
    --fp16 \
    --do_test \
    --num_train_epochs 10 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --device_ids 0 \
    --seed 42
    
# zero shot
python run.py \
    --model_name_or_path="XQ112/OpCodeBERT" \
    --dataset="CosQA" \
    --output_dir="saved_models/CosQA" \
    --do_zero \
    --num_train_epochs 10 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --device_ids 0 \
    --seed 42
```