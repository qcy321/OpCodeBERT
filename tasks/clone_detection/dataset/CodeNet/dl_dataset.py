import argparse
import ast
from multiprocessing import Pool

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def process_data(d):
    result = []
    try:
        code = ast.unparse(ast.parse(d['code']))
        result.append({"label": int(d['problem_id'][2:]), "code": [code]})
    except Exception as e:
        pass
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process Python code.")
    parser.add_argument("--num_processes", type=int, default=16,
                        help="Number of processes to use")
    parser.add_argument("--data_dir", default="./", type=str,
                        help="Directory for data output")
    parser.add_argument("--dataset", default="small", type=str,
                        help="Directory for data output")

    args = parser.parse_args()
    ds = load_dataset("windchimeran/codenet_python")
    datas = ds[args.dataset]

    with Pool(processes=args.num_processes) as pool:
        results = list(tqdm(pool.imap(process_data, datas), total=len(datas)))

    data = []
    data_index = {}
    for sub_results in results:
        for item in sub_results:
            g = item["label"]
            if g not in data_index:
                data_index[g] = len(data)
                data.append(item)
            else:
                data[data_index[g]]["code"].extend(item["code"])

    pd.DataFrame(data).to_json(f'{args.data_dir}/{args.dataset}.jsonl', orient='records', lines=True)
