import pandas as pd
import os


DEEPCOM = "/home/qiuyuanchen/Onedrive/EMSE-DeepCom/my_test"
CODE2SEQ = "/home/qiuyuanchen/Onedrive/code2seq-master/my_test"
NNGEN = "/home/qiuyuanchen/Onedrive/nngen/my_test"
MERGE = "/home/qiuyuanchen/Onedrive/my_parser/src/main/resources/merge_result"

deepcom_res = os.path.join(DEEPCOM, "results.txt")
code2seq_res = os.path.join(CODE2SEQ, "results.txt")
nngen_res = os.path.join(NNGEN, "results.txt")
merge_res = os.path.join(MERGE, "merge_results.txt")


def read_results(data, input_nlgeval, method_name):
    with open(input_nlgeval) as f:
        data["Method"].append(method_name)
        for line in f.readlines():
            if line.startswith("Bleu_1"):
                res = line.split(": ")[1].strip()
                data['BLEU-1'].append(res)
            if line.startswith("Bleu_2"):
                res = line.split(": ")[1].strip()
                data['BLEU-2'].append(res)
            if line.startswith("Bleu_3"):
                res = line.split(": ")[1].strip()
                data['BLEU-3'].append(res)
            if line.startswith("Bleu_4"):
                res = line.split(": ")[1].strip()
                data['BLEU-4'].append(res)
            if line.startswith("ROUGE_L"):
                res = line.split(": ")[1].strip()
                data['ROUGE-L'].append(res)

        return data


def main():
    data = {
        "Method": [],
        "ROUGE-L": [],
        "BLEU-1": [],
        "BLEU-2": [],
        "BLEU-3": [],
        "BLEU-4": [],
    }
    data = read_results(data, deepcom_res, "DeepCom")
    data = read_results(data, code2seq_res, "Code2Seq")
    data = read_results(data, nngen_res, "NNgen")
    data = read_results(data, merge_res, "Merge")

    data = pd.DataFrame(data)
    print(data)

    excel_path = os.path.join(MERGE, "RQ3-results.xlsx")
    data.to_excel(excel_path)


if __name__ == "__main__":
    main()
