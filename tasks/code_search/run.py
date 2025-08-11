import argparse
import json
import logging
import os
import random

import torch
from torch import nn, optim
import pandas as pd
import numpy as np
from torch.amp import GradScaler, autocast

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import get_linear_schedule_with_warmup, AutoModel, AutoTokenizer
from util import run, FunctionInf, split_task
from model import OpCodeModel

logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 nl_ids,
                 code_ids,
                 url
                 ):
        self.code_ids = code_ids
        self.nl_ids = nl_ids
        self.url = url


def convert_all(items):
    df, tokenizer, args = items
    result = []
    for i in range(df.shape[0]):
        out = convert_examples_to_features(df.iloc[i], tokenizer, args)
        result.append(out)
    return result


def convert_examples_to_features(js, tokenizer, args):
    # nl
    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 4]
    nl_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    # opcode
    opcode = js["opcode_string"]
    opcode_tokens = tokenizer.tokenize(opcode)[:args.opcode_length - 4]
    opcode_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + opcode_tokens + [
        tokenizer.sep_token]
    opcode_ids = tokenizer.convert_tokens_to_ids(opcode_tokens)
    padding_length = (args.opcode_length - len(opcode_ids))
    opcode_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(nl_ids, opcode_ids, js['url'] if "url" in js else js["retrieval_idx"])


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
        return (torch.tensor(self.examples[item].nl_ids),
                torch.tensor(self.examples[item].code_ids))


def contrastive_loss(q, k, args, temperature=0.05):
    batch_size = q.shape[0]

    sim = torch.einsum('ac,bc->ab', [q, k])

    loss = CrossEntropyLoss()(sim / temperature, torch.arange(batch_size, device=args.device))
    return loss


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

    scaler = GradScaler(enabled=args.fp16)
    last_path = f"{args.output_dir}/checkpoint-last-mrr"
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
    tr_num, tr_loss, best_mrr = 0, 0, 0
    best_mrr = args.best_mrr
    for idx in range(args.start_idx, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            nl_inputs = batch[0].to(args.device)
            code_inputs = batch[1].to(args.device)

            with autocast(device_type='cuda', enabled=args.fp16):
                q = model(nl_inputs, attention_mask=nl_inputs.ne(1))
                k = model(code_inputs, attention_mask=code_inputs.ne(1))
            loss = contrastive_loss(q, k, args, args.temperature)

            tr_loss += loss.item()
            tr_num += 1

            if (step + 1) % 100 == 0:
                logger.info(f"epoch {idx} step {step + 1} loss {round(tr_loss / tr_num, 5)}")
                tr_loss, tr_num = 0, 0

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        if not args.code_testing:
            checkpoint_prefix = 'checkpoint-last-mrr'
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
        logger.info("Evaluation results:")
        for key, value in results.items():
            logger.info("  %s = %s", key, value)

        with open(args.mrr_result, "a") as f:
            f.write(json.dumps(results) + "\n")

        # save best model
        if results['eval_mrr'] > best_mrr and not args.code_testing:
            best_mrr = results['eval_mrr']
            logger.info("  " + "*" * 20)
            logger.info("  Best mrr:%s", best_mrr)
            logger.info("  " + "*" * 20)

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_path = os.path.join(output_dir, '{}'.format('model.bin'))
            torch.save(model_to_save.state_dict(), model_path)
            logger.info("Saving model checkpoint to %s", output_dir)
            with open(f"{output_dir}/best.txt", "w") as f:
                f.write(str(best_mrr))
            logger.info(f"Saving {best_mrr} checkpoint to {output_dir}/best.txt")
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)


def evaluate(args, model, tokenizer, file_name):
    nl_dataset = TextDataset(tokenizer, args, file_name)
    nl_sampler = SequentialSampler(nl_dataset)
    nl_dataloader = DataLoader(nl_dataset, sampler=nl_sampler,
                               batch_size=args.eval_batch_size * max(1, args.n_gpu), num_workers=0)

    code_dataset = nl_dataset if args.dataset == "AdvTest" else TextDataset(tokenizer, args, args.codebase_file)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler,
                                 batch_size=args.eval_batch_size * max(1, args.n_gpu), num_workers=0)

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(nl_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size * max(1, args.n_gpu))

    model.eval()
    code_vecs, nl_vecs = [], []
    for batch in nl_dataloader:
        nl_inputs = batch[0].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs)
            nl_vecs.append(nl_vec.cpu().numpy())

    for batch in code_dataloader:
        code_inputs = batch[1].to(args.device)
        with torch.no_grad():
            code_vec = model(code_inputs)
            code_vecs.append(code_vec.cpu().numpy())

    model.train()
    code_vecs = np.concatenate(code_vecs, 0)
    nl_vecs = np.concatenate(nl_vecs, 0)

    scores = np.matmul(nl_vecs, code_vecs.T)
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

    nl_urls = [ex.url for ex in nl_dataset.examples]
    code_urls = [ex.url for ex in code_dataset.examples]
    top_k = [1, 5, 10, 50, 100]
    acc_k = [0] * len(top_k)
    ranks = []

    for url, sort_id in zip(nl_urls, sort_ids):
        rank = 0
        for idx in sort_id[:1000]:
            rank += 1
            if code_urls[idx] == url:
                for i, k in enumerate(top_k):
                    if k >= rank:
                        acc_k[i] += 1
                ranks.append(1 / rank)
                break
        else:
            ranks.append(0)

    return {
        "eval_mrr": round(float(np.mean(ranks)), 6),
        "top_k": top_k,
        "acc_k": (np.round(np.array(acc_k) / len(nl_dataset), 6)).tolist()
    }


def main(args):
    # set log
    logging.basicConfig(filename=args.log, format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    # Setup CUDA, GPU & distributed training
    device = torch.device(f"cuda:{args.device_ids[0]}" if torch.cuda.is_available() else "cpu")
    args.n_gpu = len(args.device_ids)
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    logger.warning(" device: %s, n_gpu: %s", device, args.n_gpu)
    args.device = device

    # Set seed
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    pre_last_path = f"{args.output_dir}/checkpoint-last-mrr"
    pre_best_path = f"{args.output_dir}/checkpoint-best-mrr"
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

    model = AutoModel.from_pretrained(args.model_name_or_path)
    model = OpCodeModel(model)

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
        checkpoint_prefix = 'checkpoint-last-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        if os.path.exists(output_dir):
            model.load_state_dict(torch.load(output_dir, map_location=args.device), strict=False)
        model.to(args.device)
        result = evaluate(args, model, tokenizer, args.eval_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        if os.path.exists(output_dir):
            model.load_state_dict(torch.load(output_dir, map_location=args.device), strict=False)
        model.to(args.device)
        result = evaluate(args, model, tokenizer, args.test_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

    if args.do_zero:
        model.to(args.device)
        result = evaluate(args, model, tokenizer, args.test_data_file)
        logger.info("***** zero shot results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    # parser.add_argument("--train_data_file", default="dataset/CSN/python/train.jsonl", type=str,
    #                     help="The input training data file (a json file).")
    # parser.add_argument("--eval_data_file", default="dataset/CSN/python/valid.jsonl", type=str,
    #                     help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    # parser.add_argument("--test_data_file", default="dataset/CSN/python/test.jsonl", type=str,
    #                     help="An optional input test data file to test the MRR(a josnl file).")
    # parser.add_argument("--codebase_file", default="dataset/CSN/python/codebase.jsonl", type=str,
    #                     help="An optional input test data file to codebase (a jsonl file).")
    parser.add_argument("--output_dir", type=str,
                        help="The directory where model predictions and checkpoints will be saved.")

    parser.add_argument("--dataset", type=str,
                        help='Select the dataset to use: CSN, AdvTest, or CosQA.')

    parser.add_argument('--log', type=str, default="train.log",
                        help="Path to the log file for training output.")
    parser.add_argument('--mrr_result', type=str, default="mrr_result.txt",
                        help="Path to the file where MRR results will be saved.")

    parser.add_argument("--model_name_or_path", type=str,
                        help="Path to the pre-trained model or checkpoint for weight initialization.")

    parser.add_argument("--nl_length", default=128, type=int,
                        help="Maximum sequence length for natural language inputs after tokenization.")
    parser.add_argument("--opcode_length", default=320, type=int,
                        help="Maximum sequence length for opcode inputs after tokenization.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_zero", action='store_true',
                        help="zero short task.")
    parser.add_argument("--code_testing", action='store_true',
                        help="Flag to enable code testing mode.")
    parser.add_argument("--fp16", action='store_true',
                        help="Whether to use 16-bit floating-point precision for training.")

    parser.add_argument("--cpu_cont", default=8, type=int,
                        help="cpu core count")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--temperature", default=0.05, type=float,
                        help="Temperature value controlling the randomness of model output.")
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
        "CSN": "./dataset/CSN/python",
        "AdvTest": "./dataset/AdvTest",
        "CosQA": "./dataset/CosQA"
    }
    files = {
        "CSN": ["codebase.jsonl", "train.jsonl", "test.jsonl", "valid.jsonl"],
        "AdvTest": ["valid.jsonl", "train.jsonl", "test.jsonl", "valid.jsonl"],
        "CosQA": ["code_idx_map.json", "cosqa-retrieval-train-19604.json", "cosqa-retrieval-test-500.json",
                  "cosqa-retrieval-dev-500.json"]
    }

    args.codebase_file = data_dir[args.dataset] + "/" + files[args.dataset][0]
    args.train_data_file = data_dir[args.dataset] + "/" + files[args.dataset][1]
    args.test_data_file = data_dir[args.dataset] + "/" + files[args.dataset][2]
    args.eval_data_file = data_dir[args.dataset] + "/" + files[args.dataset][3]

    gpu_count = torch.cuda.device_count()
    if args.device_ids is not None:
        device_ids = [int(id) for id in args.device_ids.split(",")]
    else:
        device_ids = [i for i in range(gpu_count)]
    args.device_ids = device_ids
    main(args)
