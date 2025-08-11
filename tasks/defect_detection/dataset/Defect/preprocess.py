import json
import tokenize
from io import StringIO, BytesIO


def remove_comments_and_docstrings(source):
    """
    Returns 'source' minus comments and docstrings.
    """
    io_obj = StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        # Remove comments:
        if token_type == tokenize.COMMENT:
            pass
        # This series of conditionals removes docstrings:
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                # This is likely a docstring; double-check we're not inside an operator:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    temp = []
    for x in out.split('\n'):
        if x.strip() != "":
            temp.append(x)
    return '\n'.join(temp)


print("***Data processing begins***")
test, train, validation = [], [], []
file_test, file_train, file_validation = 'test.jsonl', 'train.jsonl', 'validation.jsonl'
for datas, file in [[test, file_test], [train, file_train], [validation, file_validation]]:
    with open(file) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            if js["programming_language"] == "Python":
                try:
                    js["code"] = js["code"].strip()
                    code = remove_comments_and_docstrings(js["code"])
                    b = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
                    tokens = []
                    for i, t in enumerate(b):
                        if i != 0 and t.string.strip() != "":
                            tokens.append(t.string)
                    js["code_tokens"] = ' '.join(tokens)
                    datas.append(js)
                except:
                    continue
    with open(file, "w") as f:
        for data in datas:
            f.write(json.dumps(data) + '\n')

print(f"test.jsonl:{len(test)}")
print(f"validation.jsonl:{len(validation)}")
print(f"train.jsonl:{len(train)}")
print("***Data processing completed***")
