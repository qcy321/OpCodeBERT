import argparse
import multiprocessing
import os.path
import types
import ast
import dis
import logging

import pandas as pd

from util import Log, run, FunctionInf, split_task, DfData

logger = logging.getLogger(__name__)

SPECIAL_OPCODE = {
    "JUMP_FORWARD": "RESPONSE_FORWARD",
    "JUMP_ABSOLUTE": "RESPONSE_ABSOLUTE",
    "POP_JUMP_IF_FALSE": "RESPONSE_IF_FALSE",
    "POP_JUMP_IF_TRUE": "RESPONSE_IF_TRUE",
    "JUMP_IF_FALSE_OR_POP": "RESPONSE_OR_FALSE",
    "JUMP_IF_TRUE_OR_POP": "RESPONSE_OR_TRUE",
    "JUMP_IF_NOT_EXC_MATCH": "RESPONSE_EXC",
    "FOR_ITER": "RESPONSE_FOR",
    "SETUP_FINALLY": "RESPONSE_FINALLY",
    "SETUP_WITH": "RESPONSE_WITH",
    "SETUP_ASYNC_WITH": "RESPONSE_ASYNC_WITH",
    "END_ASYNC_FOR": "RESPONSE_ASYNC_FOR",
}

OPCODE_OBJECT_START = "<OPCODE_OBJECT_START>"
OPCODE_OBJECT_END = "<OPCODE_OBJECT_END>"
OPCODE_OBJECT_NAME_START = "<OPCODE_OBJECT_NAME_START>"
OPCODE_OBJECT_NAME_END = "<OPCODE_OBJECT_NAME_END>"
OPCODE_IN_OBJECT_START = "<OPCODE_IN_OBJECT_START>"
OPCODE_IN_OBJECT_END = "<OPCODE_IN_OBJECT_END>"
OPCODE_OBJECT_CONTENT_START = "<OPCODE_OBJECT_CONTENT_START>"
OPCODE_OBJECT_CONTENT_END = "<OPCODE_OBJECT_CONTENT_END>"
OPCODE_OBJECT = "<OPCODE_OBJECT>"
OPCODE_START = "<OPCODE_START>"
OPCODE_END = "<OPCODE_END>"

OPCODE_MARKERS = (
    OPCODE_OBJECT_START,
    OPCODE_OBJECT_END,
    OPCODE_OBJECT_NAME_START,
    OPCODE_OBJECT_NAME_END,
    OPCODE_IN_OBJECT_START,
    OPCODE_IN_OBJECT_END,
    OPCODE_OBJECT_CONTENT_START,
    OPCODE_OBJECT_CONTENT_END,
    OPCODE_OBJECT,
    OPCODE_END
)


def remove_blank_lines(code: str) -> list[str]:
    """Remove blank lines from text and return a list of non-blank lines."""
    return [line for line in code.split('\n') if line.strip()]


def generate_opcode_sequence(source, start=None, end=None, name=None):
    """Generate opcode sequence with nested structure."""
    response_index = {}
    index_map = {}
    seq = []
    if start is not None:
        seq.extend(start)
    if name is not None:
        seq.extend(name)
    for sou in source:
        if "RESUME".__eq__(sou.opname):
            continue
        if sou.is_jump_target:
            seq.append("<>")
            response_index[f"{len(seq) - 1}"] = sou.offset
        seq.append(sou.opname)
        if type(sou.argval) == types.CodeType:
            if sou.argval.co_name.startswith('<') and sou.argval.co_name.endswith('>'):
                object_start = OPCODE_IN_OBJECT_START
                object_end = OPCODE_IN_OBJECT_END
            else:
                object_start = OPCODE_OBJECT_START
                object_end = OPCODE_OBJECT_END
            seq.extend(generate_opcode_sequence(dis.get_instructions(sou.argval), [object_start],
                                                [OPCODE_OBJECT_CONTENT_END, object_end], [
                                                    OPCODE_OBJECT_NAME_START,
                                                    sou.argval.co_name,
                                                    OPCODE_OBJECT_NAME_END,
                                                    OPCODE_OBJECT_CONTENT_START,
                                                ]))
        else:
            if sou.opname in SPECIAL_OPCODE:
                index_map[sou.argval] = SPECIAL_OPCODE[sou.opname]
            elif sou.argrepr != '':
                seq.append(sou.argrepr)
    for key in response_index:
        seq[int(key)] = index_map.get(response_index[key], seq[int(key)])
    if end is not None:
        seq.extend(end)
    return seq


def generate_opcode_from_source(code) -> str:
    """
    Generate opcode sequence from source code.

    :param code: Source code as string
    :return: Tuple of opcode sequence and cleaned code lines
    """
    opcode = dis.Bytecode(code)
    seq = generate_opcode_sequence(opcode, [OPCODE_START], [OPCODE_END])
    return " ".join(seq)


def single_process_to_opcode_list(df: pd.DataFrame) -> pd.DataFrame:
    """Convert function code to opcode list in a single process."""
    opcode_col = "opcode_string"
    code_col = "code"
    df[opcode_col] = [""] * len(df)
    if code_col not in df.columns:
        df.rename(columns={'function': 'code', 'function_tokens': 'code_tokens'}, inplace=True)
    if 'url' not in df.columns:
        df.rename(columns={'retrieval_idx': 'url'}, inplace=True)
    rows_to_drop = []
    for idx in df.index:
        code = df.loc[idx][code_col]
        try:
            source_code = ast.unparse(ast.parse(code))
            opcode_seq = generate_opcode_from_source(source_code)
            df.at[idx, opcode_col] = opcode_seq
            df.at[idx, code_col] = source_code
        except Exception as e:
            logger.debug(f"Failed to parse function at index {idx}: {str(e)}", exc_info=True)
            rows_to_drop.append(idx)
            continue
    if rows_to_drop:
        df = df.drop(rows_to_drop)
    return df


def mult_data_processing(args, df_data: DfData, chunk_size: int = 500) -> None:
    """Process data in multiple processes and save results."""
    df_list: list[pd.DataFrame] = split_task(df_data.df, chunk_size)
    tasks: list[FunctionInf] = [FunctionInf(single_process_to_opcode_list, (df,)) for df in df_list]

    logger.info("-----Starting parallel processing-----")
    new_list: list[pd.DataFrame] = run(args.num_processes, tasks, df_data.file)

    logger.info("-----Merging data-----")
    all_df = pd.concat(new_list, ignore_index=True)

    logger.info(f"-----Data volume ({len(all_df)})-----")

    logger.info("-----Saving data-----")
    all_df.to_json(f"{args.data_dir[args.dataset]}/{df_data.file}", orient='records',
                   indent=None if df_data.file.endswith(".jsonl") else 4,
                   lines=True if df_data.file.endswith(".jsonl") else False)

    logger.info("-----Save completed-----")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_processes", type=int, default=16,
                        help='Number of processes executed')
    parser.add_argument("--dataset", type=str, default="CodeNet",
                        help='Select dataset')

    args = parser.parse_args()

    args.data_dir = {
        "CodeNet": "./dataset/CodeNet",
    }
    files = {
        "CodeNet": ["train.jsonl", "test.jsonl", "valid.jsonl"],
    }
    for file in files[args.dataset]:
        if not os.path.exists(args.data_dir[args.dataset] + "/" + file):
            raise f"{args.data_dir[args.dataset]}/{file}，does not exist"
    for file in files[args.dataset]:
        mult_data_processing(args, DfData(
            pd.read_json(args.data_dir[args.dataset] + "/" + file,
                         lines=True if file.endswith(".jsonl") else False),
            file))


if __name__ == '__main__':
    Log(logging.INFO)
    multiprocessing.freeze_support()
    main()
