from tqdm import tqdm
import pandas as pd

dataset = "small"
df = pd.read_json(f"{dataset}.jsonl", lines=True)
num = 1
clos = ["index", "label", "code"]
new_df_train = pd.DataFrame(columns=clos)
new_df_valid = pd.DataFrame(columns=clos)
new_df_test = pd.DataFrame(columns=clos)
for start, end, data, file in (
        (0, 2266, new_df_train, "train"), (2266, 2549, new_df_valid, "valid"), (2549, 2833, new_df_test, "test")):
    for i in tqdm(range(start, end), desc=file):
        label = df.loc[i]['label']
        code_list = df.loc[i]['code'][:200]
        if len(code_list) <= 10:
            continue
        for code in code_list:
            data.loc[len(data)] = {"index": num, "label": label, "code": code}
            num += 1

    data = data.sample(frac=1).reset_index(drop=True)
    print(f"-----{file} Data volume ({len(data)})-----")
    data.to_json(f"{file}.jsonl", orient="records", lines=True)
