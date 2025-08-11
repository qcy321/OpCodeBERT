# Project Usage Guide

## 1. Process Data

Run the following command to perform data preprocessing:
```bash
cd dataset/Defect

python preprocess.py

cd ../../

python data_processing.py
```

## 2. Run the Main Program
After data processing is complete, execute the main program:
```bash
python run.py \
        --do_train \
        --do_test \
        --fp16 \
        --model_name_or_path="XQ112/OpCodeBERT" \
        --dataset "Defect" \
        --output_dir="saved_models/Defect" \
        --device_ids 0 \
        --cpu_cont 10 \
        --train_batch_size 16 \
        --eval_batch_size 32 \
        --num_train_epochs 10 \
        --seed 42
```

After the run is complete, the model is saved in the ```saved_models``` directory