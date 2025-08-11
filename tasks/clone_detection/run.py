import argparse
import json
import logging
import os

import pandas as pd
import torch
import random
import numpy as np
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from model import OpCodeBERTCloneDetection
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup, AutoModel, AutoTokenizer)
from util import run, FunctionInf, split_task

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 code_ids,
                 index,
                 label
                 ):
        self.code_ids = code_ids
        self.index = index
        self.label = label


def convert_all(items):
    df, tokenizer, args = items
    result = []
    method = convert_examples_to_features_op if args.do_opcode else convert_examples_to_features_code
    for i in range(df.shape[0]):
        out = method(df.iloc[i], tokenizer, args)
        result.append(out)
    return result


def convert_examples_to_features_op(js, tokenizer: RobertaTokenizer, args):
    # opcode
    code = js["opcode_string"]
    code_tokens = tokenizer.tokenize(code)[:args.opcode_length - 4]
    code_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.opcode_length - len(code_tokens)
    code_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_ids, js["index"], js["label"])


def convert_examples_to_features_code(js, tokenizer: RobertaTokenizer, args):
    # opcode
    code = js["code"]
    code_tokens = tokenizer.tokenize(code)[:args.code_length - 4]
    code_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_tokens)
    code_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_ids, js["index"], js["label"])


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
        self.label_examples = {}
        for e in self.examples:
            if e.label not in self.label_examples:
                self.label_examples[e.label] = []
            self.label_examples[e.label].append(e)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        label = self.examples[item].label
        index = self.examples[item].index
        labels = list(self.label_examples)
        labels.remove(label)
        while True:
            shuffle_example = random.sample(self.label_examples[label], 1)[0]
            if shuffle_example.index != index:
                p_example = shuffle_example
                break
        n_example = random.sample(self.label_examples[random.sample(labels, 1)[0]], 1)[0]
        return ((torch.tensor(self.examples[item].code_ids),
                 torch.tensor(p_example.code_ids),
                 torch.tensor(n_example.code_ids)),
                torch.tensor(label))


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer):
    """ Train the model """
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size * max(1, args.n_gpu), num_workers=0)

    args.max_steps = args.num_train_epochs * len(train_dataloader)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * 0.1,
                                                num_training_steps=args.max_steps)

    # 初始化 GradScaler 用于 FP16
    scaler = GradScaler(enabled=args.fp16)  # enabled=args.fp16 确保 FP16 只在需要时生效

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size = %d", args.train_batch_size * max(1, args.n_gpu))
    logger.info("  Total optimization steps = %d", args.max_steps)

    tr_num, tr_loss, best_map = 0, 0, 0

    model.zero_grad()
    model.train()
    for idx in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with autocast(device_type='cuda', enabled=args.fp16):
                inputs = [item.to(args.device) for item in batch[0]]
                label = batch[1].to(args.device)

                loss, vec = model(inputs, label)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            # report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % 10 == 0:
                cur_loss = round(tr_loss / tr_num, 5)

                logger.info("epoch {} step {} loss {}".format(idx, (step + 1), cur_loss))
                tr_loss = 0
                tr_num = 0

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        if not args.code_testing:
            checkpoint_prefix = 'checkpoint-last-map'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_file = os.path.join(output_dir, '{}'.format('model.bin'))
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), model_file)
            logger.info("Saving model checkpoint to %s", output_dir)
            with open(f"{output_dir}/idx.txt", "w") as f:
                f.write(str(idx))
            logger.info(f"Saving {idx} checkpoint to {output_dir}/idx.txt")
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)

        results = evaluate(args, model, tokenizer, args.eval_data_file)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value, 4))

        with open(args.map_result, "a") as f:
            f.write(json.dumps(results) + "\n")

        if results['eval_map'] > best_map and not args.code_testing:
            best_map = results['eval_map']
            logger.info("  " + "*" * 20)
            logger.info("  Best map:%s", round(best_map, 4))
            logger.info("  " + "*" * 20)

            checkpoint_prefix = 'checkpoint-best-map'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, file_name):
    """ Evaluate the model """
    eval_dataset = TextDataset(tokenizer, args, file_name)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size * max(1, args.n_gpu), num_workers=0)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    vecs = []
    labels = []

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs = [item.to(args.device) for item in batch[0]]
        label = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, vec = model(inputs, label)
            eval_loss += lm_loss.mean().item()
            vecs.append(vec.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    vecs = np.concatenate(vecs, 0)
    labels = np.concatenate(labels, 0)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    scores = np.matmul(vecs, vecs.T)
    dic = {}
    for i in range(scores.shape[0]):
        scores[i, i] = -1000000
        if int(labels[i]) not in dic:
            dic[int(labels[i])] = -1
        dic[int(labels[i])] += 1
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]
    MAP = []
    for i in range(scores.shape[0]):
        cont = 0
        label = int(labels[i])
        Avep = []
        for j in range(dic[label]):
            index = sort_ids[i, j]
            if int(labels[index]) == label:
                Avep.append((len(Avep) + 1) / (j + 1))
        MAP.append(sum(Avep) / dic[label])

    result = {
        "eval_loss": float(perplexity),
        "eval_map": float(np.mean(MAP))
    }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./saved_models/CodeNet", type=str,
                        help="The directory where model predictions and checkpoints will be saved.")
    parser.add_argument("--dataset", default="CodeNet", type=str,
                        help="Select the dataset to use: CodeNet.")

    # parser.add_argument("--train_data_file", default="dataset/CodeNet/train.jsonl", type=str,
    #                     help="The input training data file (a json file).")
    # parser.add_argument("--eval_data_file", default="dataset/CodeNet/valid.jsonl", type=str,
    #                     help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    # parser.add_argument("--test_data_file", default="dataset/CodeNet/test.jsonl", type=str,
    #                     help="An optional input test data file to test the MRR(a josnl file).")

    parser.add_argument("--model_name_or_path", default="XQ112/OpCodeBERT", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument('--log', type=str, default="train.log",
                        help="Path to the log file for training output.")
    parser.add_argument('--map_result', type=str, default="map_result.txt",
                        help="Path to the file where MAP results will be saved.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_opcode", action='store_true',
                        help="Whether to use opcode.")
    parser.add_argument("--code_testing", action='store_true',
                        help="Flag to enable code testing mode.")
    parser.add_argument("--fp16", action='store_true',
                        help="Whether to use 16-bit floating-point precision for training.")

    parser.add_argument("--opcode_length", default=400, type=int,
                        help="Maximum sequence length for opcode inputs after tokenization.")

    parser.add_argument("--cpu_cont", default=8, type=int,
                        help="cpu core count")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
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
        "CodeNet": "./dataset/CodeNet",
    }
    files = {
        "CodeNet": ["train.jsonl", "test.jsonl", "valid.jsonl"],
    }

    args.train_data_file = data_dir[args.dataset] + "/" + files[args.dataset][0]
    args.test_data_file = data_dir[args.dataset] + "/" + files[args.dataset][1]
    args.eval_data_file = data_dir[args.dataset] + "/" + files[args.dataset][2]

    # set log
    logging.basicConfig(filename=args.log, format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    gpu_count = torch.cuda.device_count()

    if args.device_ids is not None:
        device_ids = [int(id) for id in args.device_ids.split(",")]
    else:
        device_ids = [i for i in range(gpu_count)]
    args.device_ids = device_ids

    # Setup CUDA, GPU & distributed training
    device = torch.device(f"cuda:{args.device_ids[0]}" if torch.cuda.is_available() else "cpu")
    args.n_gpu = len(args.device_ids)
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    args.device = device

    # Set seed
    set_seed(args.seed)

    pre_last_path = f"{args.output_dir}/checkpoint-last-map"
    pre_best_path = f"{args.output_dir}/checkpoint-best-map"
    idx_file = f"{pre_last_path}/idx.txt"
    best_file = f"{pre_best_path}/best.txt"
    args.start_idx = 0
    args.best_mrr = 0.0

    if os.path.exists(idx_file):
        with open(idx_file, "r") as f:
            args.start_idx = int(f.read()) + 1
    if os.path.exists(best_file):
        with open(best_file, "r") as f:
            args.best_mrr = float(f.read())

    # build model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = OpCodeBERTCloneDetection.from_pretrained(args.model_name_or_path)

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
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-map/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer, args.eval_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key] * 100 if "map" in key else result[key], 2)))

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-map/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        if os.path.exists(output_dir):
            model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer, args.test_data_file)
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key] * 100 if "map" in key else result[key], 2)))


if __name__ == "__main__":
    main()
