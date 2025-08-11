import argparse
import json
import logging
import os
import random

import torch
from torch.amp import GradScaler, autocast
from torch import nn, optim
import pandas as pd
import numpy as np
from model import OpCodeBERTClassification
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, Sampler
from transformers import (get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, RobertaTokenizer)
from util import run, FunctionInf, split_task

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 code_ids,
                 url,
                 labels
                 ):
        self.code_ids = code_ids
        self.url = url
        self.labels = labels


def convert_all(items):
    df, tokenizer, args = items
    result = []
    for i in range(df.shape[0]):
        out = convert_examples_to_features(df.iloc[i], tokenizer, args)
        result.append(out)
    return result


def convert_examples_to_features(js, tokenizer: RobertaTokenizer, args):
    # opcode
    code = js["opcode_string"]
    code_tokens = tokenizer.tokenize(code)[:args.opcode_length - 4]
    code_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.opcode_length - len(code_tokens)
    code_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_ids, js['url'] if "url" in js else js["retrieval_idx"], js["label"])


def get_cache_file_path(args, prefix):
    cache_file = args.output_dir + '/' + prefix + '_' + args.dataset
    suffix = '.pt'
    return cache_file + "_codetest_" + suffix if args.code_testing else cache_file + suffix


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.args = args
        prefix = os.path.splitext(os.path.basename(file_path))[0]
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cache_file = get_cache_file_path(args, prefix)
        if not os.path.exists(cache_file):
            self.examples = []
            if args.code_testing:
                df = pd.read_json(file_path, lines=True if file_path.endswith(".jsonl") else False).head(args.read_size)
            else:
                df = pd.read_json(file_path, lines=True if file_path.endswith(".jsonl") else False)
            tasks: list[FunctionInf] = []
            data_list: list = split_task(df, 500)
            for da in data_list:
                tasks.append(FunctionInf(convert_all, ((da, tokenizer, args),)))
            new_list: list = run(args.cpu_cont, tasks, prefix)
            for li in new_list:
                self.examples.extend(li)
            torch.save(self.examples, cache_file)
        if os.path.exists(cache_file):
            self.examples = torch.load(cache_file, weights_only=False)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return (torch.tensor(self.examples[item].code_ids),
                torch.tensor(self.examples[item].labels))


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer):
    """ Train the model """
    # get training dataset
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size * max(1, args.n_gpu), num_workers=0)

    num_training_steps = len(train_dataloader) * args.num_train_epochs
    # get optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=num_training_steps)

    # Initialize GradScaler for FP16
    scaler = GradScaler(enabled=args.fp16)  # enabled=args.fp16 ensures FP16 is only enabled when needed

    # Loading saved optimizers and schedulers
    last_path = f"{args.output_dir}/checkpoint-last-F1"
    optimizer_last = os.path.join(last_path, 'optimizer.pt')
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    scheduler_last = os.path.join(last_path, 'scheduler.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    # multi-gpu training (should be after apex fp16 initialization)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size  = %d", args.train_batch_size * max(1, args.n_gpu))
    logger.info("  per optimization steps = %d", len(train_dataloader))
    logger.info("  Total optimization steps = %d", num_training_steps)

    model.zero_grad()

    model.train()
    tr_num, tr_loss, F1 = 0, 0, 0
    F1 = args.F1
    for idx in range(args.start_idx, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with autocast(device_type='cuda', enabled=args.fp16):
                code_inputs = batch[0].to(args.device)
                labels = batch[1].to(args.device)

                # get outs
                outs = model(inputs=code_inputs, labels=labels)

            # calculate loss
            loss = outs[0]
            if args.n_gpu > 1:
                loss = loss.mean()
            # report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % 10 == 0:
                cur_loss = round(tr_loss / tr_num, 5)

                logger.info("epoch {} step {} loss {}".format(idx, (step + 1), cur_loss))
                tr_loss = 0
                tr_num = 0

            # backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        if not args.code_testing:
            checkpoint_prefix = 'checkpoint-last-F1'
            output_dir_last = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir_last):
                os.makedirs(output_dir_last)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_path = os.path.join(output_dir_last, '{}'.format('model.bin'))
            torch.save(model_to_save.state_dict(), model_path)
            logger.info("Saving model checkpoint to %s", output_dir_last)
            with open(f"{output_dir_last}/idx.txt", "w") as f:
                f.write(str(idx))
            logger.info(f"Saving {idx} checkpoint to {output_dir_last}/idx.txt")
            torch.save(optimizer.state_dict(), os.path.join(output_dir_last, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir_last, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir_last)

        # evaluate
        results = evaluate(args, model, tokenizer, args.eval_data_file)
        logger.info("### Eval ###")
        for key, value in results.items():
            logger.info("  %s = %s", key, value)

        with open(args.F1_result, "a") as f:
            f.write(json.dumps(results) + "\n")

        # save best model
        if results['F1'] > F1 and not args.code_testing:
            F1 = results['F1']
            logger.info("  " + "*" * 20)
            logger.info("  Best F1:%s", F1)
            logger.info("  " + "*" * 20)

            checkpoint_prefix = 'checkpoint-best-F1'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_path = os.path.join(output_dir, '{}'.format('model.bin'))
            torch.save(model_to_save.state_dict(), model_path)
            logger.info("Saving model checkpoint to %s", output_dir)
            with open(f"{output_dir}/best.txt", "w") as f:
                f.write(str(F1))
            logger.info(f"Saving {F1} checkpoint to {output_dir}/best.txt")
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)


def evaluate(args, model, tokenizer, file_name):
    code_dataset = TextDataset(tokenizer, args, file_name)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler,
                                 batch_size=args.eval_batch_size * max(1, args.n_gpu), num_workers=0)

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size * max(1, args.n_gpu))

    model.eval()
    vecs = []
    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)
        labels = batch[1].to(args.device)
        with torch.no_grad():
            vec = model(inputs=code_inputs, labels=labels)[1]
            vecs.append(vec.cpu().numpy())

    model.train()

    vecs = np.concatenate(vecs, 0)

    label_lists = []

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for example in code_dataset.examples:
        label_lists.append(example.labels)

    for i, label in enumerate(label_lists):
        if np.argmax(vecs[i]) == label:
            if label == 1:
                TP += 1
            else:
                TN += 1
        else:
            if label == 1:
                FN += 1
            else:
                FP += 1
    logger.info(f"TP:{TP},FP:{FP},FN:{FN},TN:{TN}")
    accuracy = (TP + TN) / len(label_lists) if len(label_lists) > 0 else 0
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    F1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    result = {}
    result["accuracy"] = round(accuracy, 6)
    result["precision"] = round(precision, 6)
    result["Recall"] = round(recall, 6)
    result["F1"] = round(F1, 6)

    return result


def main(args):
    logging.basicConfig(filename=args.log, format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    # Setup CUDA, GPU & distributed training
    device = torch.device(f"cuda:{args.device_ids[0]}" if torch.cuda.is_available() else "cpu")
    args.n_gpu = len(args.device_ids)
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    logger.warning(" device: %s, n_gpu: %s",
                   device, args.n_gpu)
    args.device = device

    # Set seed
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    pre_last_path = f"{args.output_dir}/checkpoint-last-F1"
    pre_best_path = f"{args.output_dir}/checkpoint-best-F1"
    idx_file = f"{pre_last_path}/idx.txt"
    best_file = f"{pre_best_path}/best.txt"
    args.start_idx = 0
    args.F1 = 0.0

    if os.path.exists(idx_file):
        with open(idx_file, "r") as f:
            args.start_idx = int(f.read()) + 1
    if os.path.exists(best_file):
        with open(best_file, "r") as f:
            args.F1 = float(f.read())

    model = OpCodeBERTClassification.from_pretrained(args.model_name_or_path)

    logger.info("Training/evaluation parameters %s", args)

    if args.code_testing:
        logger.info("****   ****************************   ****")
        logger.info("****   Code testing mode is enabled   ****")
        logger.info("****   ****************************   ****")
        args.train_batch_size = 2

    # Training
    if args.do_train:
        if args.start_idx != 0:
            logger.info(f"Continue from where we left off last time epoch {args.start_idx}")
            model.load_state_dict(torch.load(f"{pre_last_path}/model.bin"), strict=False)
        model.to(args.device)
        train(args, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-last-F1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        if os.path.exists(output_dir):
            model.load_state_dict(torch.load(output_dir, map_location=args.device), strict=False)
        model.to(args.device)
        result = evaluate(args, model, tokenizer, args.eval_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-F1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        if os.path.exists(output_dir):
            model.load_state_dict(torch.load(output_dir, map_location=args.device), strict=False)
        model.to(args.device)
        result = evaluate(args, model, tokenizer, args.test_data_file)
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output_dir", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # parser.add_argument("--train_data_file", default="dataset/Defect/train.jsonl", type=str,
    #                     help="The input training data file (a json file).")
    # parser.add_argument("--eval_data_file", default="dataset/Defect/validation.jsonl", type=str,
    #                     help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    # parser.add_argument("--test_data_file", default="dataset/Defect/test.jsonl", type=str,
    #                     help="An optional input test data file to test the MRR(a josnl file).")

    parser.add_argument("--dataset", default="Defect", type=str,
                        help='Select the dataset to use:Defect')

    parser.add_argument('--log', type=str, default="train.log",
                        help="Path to the log file for training output.")
    parser.add_argument('--F1_result', type=str, default="F1_result.txt",
                        help="Path to the file where F1 results will be saved.")

    parser.add_argument("--model_name_or_path", type=str,
                        help="Path to the pre-trained model or checkpoint for weight initialization.")

    parser.add_argument("--opcode_length", default=384, type=int,
                        help="Maximum sequence length for opcode inputs after tokenization.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--code_testing", action='store_true',
                        help="Flag to enable code testing mode.")
    parser.add_argument("--fp16", action='store_true',
                        help="Whether to use 16-bit floating-point precision for training.")

    parser.add_argument("--cpu_cont", default=8, type=int,
                        help="cpu core count.")

    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Maximum norm for gradient clipping.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--read_size', type=int, default=200,
                        help="Size limit for reading or testing data.")
    parser.add_argument('--device_ids', type=str, default=None,
                        help="Comma-separated list of device IDs for multi-GPU training.")

    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility in initialization and training.")

    # print arguments
    args = parser.parse_args()

    data_dir = {
        "Defect": "./dataset/Defect"
    }
    files = {
        "Defect": ["train.jsonl", "test.jsonl", "validation.jsonl"]
    }

    args.train_data_file = data_dir[args.dataset] + "/" + files[args.dataset][0]
    args.test_data_file = data_dir[args.dataset] + "/" + files[args.dataset][1]
    args.eval_data_file = data_dir[args.dataset] + "/" + files[args.dataset][2]

    gpu_count = torch.cuda.device_count()
    if args.device_ids is not None:
        device_ids = [int(id) for id in args.device_ids.split(",")]
    else:
        device_ids = [i for i in range(gpu_count)]
    args.device_ids = device_ids
    main(args)
