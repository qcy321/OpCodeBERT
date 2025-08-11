# Project Usage Guide


## 1. Process Data

Run the following command to perform data preprocessing:

```bash
cd dataset/CodeNet

python dl_dataset.py

python processing.py

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
        --dataset "CodeNet" \
        --cpu_cont 10 \
        --device_ids 0 \
        --train_batch_size 64 \
        --eval_batch_size 64 \
        --num_train_epochs 4 \
        --seed 42
```

After the run is complete, the model is saved in the ```saved_models``` directory