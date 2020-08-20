import os
import pandas as pd

DATA_DIR = "/home/qiuyuanchen/Onedrive/my_parser/src/main/resources/merge_data"
DEV_DIR = "/home/qiuyuanchen/Onedrive/my_parser/src/main/resources/dev"
# input
# what
what_reference = os.path.join(DATA_DIR, "what(deepcom)", "test.token.nl")
what_hypothesis = os.path.join(DATA_DIR, "what(deepcom)", "translate")
what_code = os.path.join(DATA_DIR, "what(deepcom)", "test.token.code")
# why
why_reference = os.path.join(DATA_DIR, "why(nngen)", "test.token.nl")
why_hypothesis = os.path.join(DATA_DIR, "why(nngen)", "why.nl")
why_code = os.path.join(DATA_DIR, "why(nngen)", "why.code")
# how_to_use
how_to_use_reference = os.path.join(
    DATA_DIR, "how-to-use(code2seq)", "ref.txt")
how_to_use_hypothesis = os.path.join(
    DATA_DIR, "how-to-use(code2seq)", "pred.txt")
how_to_use_code = os.path.join(DATA_DIR, "how-to-use(code2seq)", "code.txt")
# how_it_is_done
how_it_is_done_reference = os.path.join(
    DATA_DIR, "how-it-is-done(nngen)", "test.token.nl")
how_it_is_done_hypothesis = os.path.join(
    DATA_DIR, "how-it-is-done(nngen)", "how-it-is-done.nl")
how_it_is_done_code = os.path.join(
    DATA_DIR, "how-it-is-done(nngen)", "how-it-is-done.code")
# property
property_reference = os.path.join(
    DATA_DIR, "property(deepcom)", "test.token.nl")
property_hypothesis = os.path.join(DATA_DIR, "property(deepcom)", "translate")
property_code = os.path.join(DATA_DIR, "property(deepcom)", "test.token.code")
# others
others_reference = os.path.join(DATA_DIR, "others(nngen)", "test.token.nl")
others_hypothesis = os.path.join(DATA_DIR, "others(nngen)", "others.nl")
others_code = os.path.join(DATA_DIR, "others(nngen)", "others.code")

# output
merge_reference = os.path.join(DATA_DIR, "merge_reference.txt")
merge_hypothesis = os.path.join(DATA_DIR, "merge_hypothesis.txt")
merge_code = os.path.join(DATA_DIR, "merge_code.txt")
merge_category = os.path.join(DATA_DIR, "merge_category.txt")


def merge():
    with open(merge_reference, "w") as f:
        total_reference = 0
        with open(what_reference) as data:
            for line in data.readlines():
                f.write(line)
                total_reference += 1
        with open(why_reference) as data:
            for line in data.readlines():
                f.write(line)
                total_reference += 1
        with open(how_to_use_reference) as data:
            for line in data.readlines():
                f.write(line)
                total_reference += 1
        with open(how_it_is_done_reference) as data:
            for line in data.readlines():
                f.write(line)
                total_reference += 1
        with open(property_reference) as data:
            for line in data.readlines():
                f.write(line)
                total_reference += 1
        with open(others_reference) as data:
            for line in data.readlines():
                f.write(line)
                total_reference += 1
        # assert total_reference == 5000, "wrong number"+str(total_reference)

    with open(merge_hypothesis, "w") as f:
        total_hypothesis = 0
        with open(what_hypothesis) as data:
            for line in data.readlines():
                f.write(line)
                total_hypothesis += 1
        with open(why_hypothesis) as data:
            for line in data.readlines():
                f.write(line)
                total_hypothesis += 1
        with open(how_to_use_hypothesis) as data:
            for line in data.readlines():
                f.write(line)
                total_hypothesis += 1
        with open(how_it_is_done_hypothesis) as data:
            for line in data.readlines():
                f.write(line)
                total_hypothesis += 1
        with open(property_hypothesis) as data:
            for line in data.readlines():
                f.write(line)
                total_hypothesis += 1
        with open(others_hypothesis) as data:
            for line in data.readlines():
                f.write(line)
                total_hypothesis += 1
        # assert total_hypothesis == 5000, "wrong number"+str(total_hypothesis)

    with open(merge_code, "w") as f:
        total_code = 0
        with open(what_code) as data:
            for line in data.readlines():
                f.write(line)
                total_code += 1
        with open(why_code) as data:
            for line in data.readlines():
                f.write(line)
                total_code += 1
        with open(how_to_use_code) as data:
            for line in data.readlines():
                f.write(line)
                total_code += 1
        with open(how_it_is_done_code) as data:
            for line in data.readlines():
                f.write(line)
                total_code += 1
        with open(property_code) as data:
            for line in data.readlines():
                f.write(line)
                total_code += 1
        with open(others_code) as data:
            for line in data.readlines():
                f.write(line)
                total_code += 1
    assert total_code == total_reference
    print(total_code)
    print("代码补齐搞定")


def output_category():
    categories = []
    with open(what_reference) as data:
        category = "what"
        num = len(data.readlines())
        categories += [category] * num
    with open(why_reference) as data:
        category = "why"
        num = len(data.readlines())
        categories += [category] * num
    with open(how_to_use_reference) as data:
        category = "how_to_use"
        num = len(data.readlines())
        categories += [category] * num
    with open(how_it_is_done_reference) as data:
        category = "how_it_is_done"
        num = len(data.readlines())
        categories += [category] * num
    with open(property_reference) as data:
        category = "property"
        num = len(data.readlines())
        categories += [category] * num
    with open(others_reference) as data:
        category = "others"
        num = len(data.readlines())
        categories += [category] * num

    with open(merge_category, "w") as f:
        for category in categories:
            f.write(category)
            f.write("\n")
    if os.path.exists(merge_category):
        with open(merge_category) as f:
            length = len(f.readlines())
            assert length == len(categories), str(
                length) + "!=" + str(len(categories))
            if length == len(categories):
                print("成功，一共{}条数据".format(length))
    print("类别分布")
    count = pd.Series(categories).value_counts()
    print(count)


def evaluate():
    # 最优效果
    command = "nlg-eval --references {} --hypothesis {} --no-skipthoughts --no-glove".format(
        merge_reference,
        merge_hypothesis
    )
    os.system(command)


def code2seq_data():
    # 因为预处理的原因，需要补全code2seq的数据
    with open(how_to_use_reference) as f:
        data = list(f.readlines())

    code_file = os.path.join(DATA_DIR, "how-to-use(code2seq)", "test.source")
    nl_file = os.path.join(DATA_DIR, "how-to-use(code2seq)", "test.token.nl")
    with open(code_file) as f:
        code = list(f.readlines())
    with open(nl_file) as f:
        nl = list(f.readlines())
    assert len(code) == len(nl)
    print("Code2Seq抽取数量：")
    print(len(data))
    print("实际标记数量：")
    print(len(nl))
    print("重复的")
    print(len(nl) - len(list(set(nl))))

    # 开始查找
    nl = [s.split(" .")[0] + "\n" if "." in s else s for s in nl]
    nl = [s.split(" ?")[0] + "\n" if "?" in s else s for s in nl]
    nl = [s.split(" !")[0] + "\n" if "!" in s else s for s in nl]
    index = []
    for comment in data:
        if comment in nl:
            line_num = nl.index(comment)
            index.append(line_num)
    print(len(index))

    with open(how_to_use_code, "w") as f:
        for num in index:
            f.write(code[num])
    print("补全成功")


def split_dev():
    # 不需要了
    with open(merge_code) as f:
        code = list(f.readlines())
    with open(merge_reference) as f:
        nl = list(f.readlines())
    print(len(code))
    # output
    train_code = code[:3000]
    train_nl = code[:3000]
    test_code = code[3000:]
    test_nl = code[3000:]
    dev_train_code = os.path.join(DEV_DIR, "dev_train_code.txt")
    dev_train_nl = os.path.join(DEV_DIR, "dev_train_nl.txt")
    dev_test_code = os.path.join(DEV_DIR, "dev_test_code.txt")
    dev_test_nl = os.path.join(DEV_DIR, "dev_test_nl.txt")

    def write(data, path):
        with open(path, "w") as f:
            for i in data:
                f.write(i)

    write(train_code, dev_train_code)
    write(train_nl, dev_train_nl)
    write(test_code, dev_test_code)
    write(test_nl, dev_test_nl)

    print("finished")


def label_5000():
    label_path = os.path.join(DEV_DIR, "test.csv")
    init_5000 = os.path.join(DEV_DIR, "label_5000.txt")
    data = pd.read_csv(label_path, index_col=0)
    print(data.columns)
    categories = data[' category']
    with open(init_5000, "w") as f:
        for category in categories:
            f.write(category)
            f.write("\n")

def main():
    # merge()
    # evaluate()
    # split_dev()
    output_category()


if __name__ == "__main__":
    # main()
    # code2seq_data()
    label_5000()
