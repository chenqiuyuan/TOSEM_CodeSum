"""Code Comment Category Prediction Model

Usage:
cccpm.py clf --classifier=<cls>
cccpm.py eval --classifier=<cls>
cccpm.py -h | --help
cccpm.py -v | --version


Options:
--classifier=<clf>            Classifier types.
-h --help                       Show this screen.
-v --version                    Show version number.

"""


import logging
import os
import string

import lightgbm as lgb
import numpy as np
import pandas as pd
from docopt import docopt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

logging.basicConfig(format="%(levelname)s| %(message)s", level=logging.INFO)

DEV_DIR = "/home/qiuyuanchen/Onedrive/my_parser/src/main/resources/dev"
DATA_DIR = "/home/qiuyuanchen/Onedrive/my_parser/src/main/resources/merge_data"
TEST_DIR = "/home/qiuyuanchen/data/huxing_data/test"
EMPIRICAL_DIR = "/home/qiuyuanchen/data/huxing_data/empirical"


merge_code = os.path.join(DATA_DIR, "merge_code.txt")
merge_category = os.path.join(DATA_DIR, "merge_category.txt")
unlabel_code = os.path.join(DEV_DIR, "test.source")


EMPIRICAL_CODE = os.path.join(EMPIRICAL_DIR, "code.txt")
# EMPIRICAL_CODE = os.path.join(EMPIRICAL_DIR, "code_5000.txt")
EMPIRICAL_LABEL = os.path.join(EMPIRICAL_DIR, "my_label.txt")
# EMPIRICAL_LABEL = os.path.join(EMPIRICAL_DIR, "label_5000.txt")
TEST_CODE = os.path.join(TEST_DIR, "code_20000.txt")

# output
PRED_LABEL = os.path.join(TEST_DIR, "label_20000.txt")


class What2WriteClassifier:
    text = []
    category = []
    @classmethod
    def __init__(cls):
        # super().__init__()
        pass

    @classmethod
    def read_data(cls, text_path, category_path):
        logging.info("STEP: read data")
        with open(text_path) as f:
            for line in tqdm(f.readlines()):
                cls.text.append(line.strip())
        with open(category_path) as f:
            for line in tqdm(f.readlines()):
                cls.category.append(line.strip())
        logging.debug(pd.Series(cls.category).value_counts())

    @classmethod
    def preprocess_text(cls):
        logging.debug(cls.text[:3])
        logging.debug(cls.category[:3])
        processed_text = []
        for s in cls.text:
            s = "".join([c for c in s if c not in string.punctuation])
            s = " ".join(s.split())
            processed_text.append(s)

        cls.text = processed_text
        logging.debug(cls.text[:5])

    @staticmethod
    def preprocess(text):
        logging.debug(text[:3])
        processed_text = []
        for s in text:
            s = "".join([c for c in s if c not in string.punctuation])
            s = " ".join(s.split())
            processed_text.append(s)

        return processed_text

    @classmethod
    def feature(cls):
        logging.info("numerical feature")
        vectorizer = TfidfVectorizer(use_idf=False)
        cls.X = vectorizer.fit_transform(cls.text)
        feature_number = len(vectorizer.get_feature_names())
        logging.debug("token数量：" + str(feature_number))
        logging.debug("矩阵形状")
        logging.debug(cls.X.shape)
        cls.vectorizer = vectorizer

        # 将类别转换为数字
        le = preprocessing.LabelEncoder()
        # 它是按照字母序来的
        le.fit(["what", "why", "how_to_use",
                "how_it_is_done", "property", "others"])
        cls.y = le.transform(cls.category)
        cls.classes = le.classes_  # 字母序的类别列表
        logging.debug(cls.y)
        logging.debug(le.inverse_transform([0, 1, 2, 3, 4, 5]))
        cls.le = le

    @staticmethod
    def _feature_transform(text, label):
        vectorizer = TfidfVectorizer(use_idf=False)
        X = vectorizer.fit_transform(text)

        # label_encoder将类别转换为数字, 它是按照字母序来的
        # inverse_transform可以将其转换
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(["what", "why", "how_to_use",
                           "how_it_is_done", "property", "others"])
        y = label_encoder.transform(label)
        classes = label_encoder.classes_
        logging.debug(label_encoder.inverse_transform([0, 1, 2, 3, 4, 5]))

        return X, y, label_encoder, classes

    @staticmethod
    def _feature_X(text):
        vectorizer = TfidfVectorizer(use_idf=False)
        X = vectorizer.fit_transform(text)

        return X, vectorizer

    @staticmethod
    def _feature_y(labels, label_names=["what", "why", "how_to_use",
                                        "how_it_is_done", "property", "others"]):
        # label_encoder将类别转换为数字, 它是按照字母序来的
        # inverse_transform可以将其转换
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(label_names)
        y = label_encoder.transform(labels)
        classes = label_encoder.classes_
        logging.debug(label_encoder.inverse_transform([0, 1, 2, 3, 4, 5]))

        return y, label_encoder, classes

    @staticmethod
    def _train_naive(X, y):
        logging.info("naive")
        naive = MultinomialNB()
        clf = naive.fit(X, y)
        return clf

    @staticmethod
    def _train_rf(X, y):
        rf = RandomForestClassifier(n_estimators=100)
        clf = rf.fit(X, y)
        return clf

    @staticmethod
    def _train_lightgbm(X, y):
        logging.info("light GBM!")
        lgbm = lgb.LGBMClassifier()
        clf = lgbm.fit(X, y)
        return clf

    @staticmethod
    def _train_dt(X, y):
        logging.info("decision tree!")
        dt = DecisionTreeClassifier()
        clf = dt.fit(X, y)
        return clf
        
    @classmethod
    def _one_fold_evaluate(cls, X, y, classes=None, splits=5):
        skf = StratifiedKFold(n_splits=splits)
        # skf = KFold(n_splits=4)

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = cls._train_rf(X_train, y_train)
            # clf = cls._train_lightgbm(X_train, y_train)
            # clf = cls._train_naive(X_train, y_train)
            predicted = clf.predict(X_test)
            results = classification_report(
                y_test, predicted, target_names=classes)
            logging.info(results)
            break

    @classmethod
    def evaluate_classifier(cls, method='rf'):
        labels = []
        codes = []
        with open(EMPIRICAL_LABEL) as f:
            for line in tqdm(f.readlines()):
                labels.append(line.strip())
        with open(EMPIRICAL_CODE) as f:
            for line in tqdm(f.readlines()):
                codes.append(line.strip())
        codes = codes[:20000]
        assert len(codes) == len(labels), "{} != {}".format(
            str(len(codes), str(len(labels))))
        codes = cls.preprocess(codes)
        X, y, label_encoder, classes = cls._feature_transform(codes, labels)
        scorings = ['precision_micro', 'recall_micro', 'f1_micro',
                    'precision_weighted', 'recall_weighted', 'f1_weighted']
        clf = RandomForestClassifier(n_estimators=100)
        if method == "naive":
            clf = MultinomialNB()
        elif method == "dt":
            clf = DecisionTreeClassifier()
        elif method == "lgbm":
            clf = lgb.LGBMClassifier(n_estimators=100)

        scores = cross_validate(clf, X, y, n_jobs=-1,
                                cv=10, verbose=True, scoring=scorings)
        # print(scores)
        print("Method:" + method)
        print("precision micro:")
        print(scores["test_precision_micro"].mean())
        print("recall micro:")
        print(scores["test_recall_micro"].mean())
        print("f1 micro:")
        print(scores["test_f1_micro"].mean())
        print("precision weighted:")
        print(scores["test_precision_weighted"].mean())
        print("recall weighted:")
        print(scores["test_recall_weighted"].mean())
        print("f1 weighted:")
        print(scores["test_f1_weighted"].mean())
        # cls.cross_evaluate(X, y, classes, splits=10)

    @classmethod
    def build_classifier_and_predict(cls, code_file, label_file, test_code_file, clf='rf') -> list:
        raw_codes, nls = cls._read_code_nl(code_file, label_file)
        codes = cls.preprocess(raw_codes)
        X, vectorizer = cls._feature_X(codes)
        y, label_encoder, classes = cls._feature_y(nls)
        if clf == 'rf':
            clf = cls._train_rf(X, y)
        if clf == 'naive':
            clf = cls._train_naive(X, y)
        if clf == 'lgbm':
            clf = cls._train_lightgbm(X, y)
        if clf == 'dt':
            clf = cls._train_dt(X, y)
        logging.info(classes)
        # ['how_it_is_done' 'how_to_use' 'others' 'property' 'what' 'why']
        # [0, 1, 2, 3, 4, 5]

        raw_codes_test = cls._read_lines(test_code_file)
        codes_test = cls.preprocess(raw_codes_test)
        test_X = vectorizer.transform(codes_test)
        pred_y = clf.predict(test_X)
        pred_classes = label_encoder.inverse_transform(pred_y)
        logging.debug(pd.Series(pred_classes).value_counts())
        return pred_y

    @staticmethod
    def predict(clf):
        pass

    @staticmethod
    def nlg_eval():
        pass

    @staticmethod
    def _read_code_nl(code_path, nl_path):
        # 一步一步写，不要耦合到一起
        codes = []
        nls = []
        with open(code_path) as f:
            for line in f.readlines():
                codes.append(line.strip())

        with open(nl_path) as f:
            for line in f.readlines():
                nls.append(line.strip())

        return codes, nls

    @staticmethod
    def _read_lines(path) -> list:
        lines = []
        with open(path) as f:
            for line in f.readlines():
                lines.append(line.strip())
        return lines


class CompositeCodeSum:
    selected_pred = []
    selected_ref = []

    def __init__(self):
        super().__init__()


    @classmethod
    def read_deepcom(cls, pred_label, result_dir):
        pred_path = os.path.join(result_dir, "pred.txt")
        ref_path = os.path.join(result_dir, "ref.txt")
        pred = cls._read_lines(pred_path)
        ref = cls._read_lines(ref_path)

        assert len(pred) == len(pred_label), "{} != {}".format(
            str(len(pred)), str(len(pred_label)))
        for index, label in enumerate(pred_label):
            # ['how_it_is_done' 'how_to_use' 'others' 'property' 'what' 'why']
            # [0, 1, 2, 3, 4, 5]
            if label in [3, 4]:
                cls.selected_pred.append(pred[index])
                cls.selected_ref.append(ref[index])

    @classmethod
    def read_nngen(cls, pred_label, result_dir):
        pred_path = os.path.join(result_dir, "pred.txt")
        ref_path = os.path.join(result_dir, "ref.txt")
        pred = cls._read_lines(pred_path)
        ref = cls._read_lines(ref_path)

        assert len(pred) == len(pred_label), "{} != {}".format(
            str(len(pred)), str(len(pred_label)))
        for index, label in enumerate(pred_label):
            # ['how_it_is_done' 'how_to_use' 'others' 'property' 'what' 'why']
            # [0, 1, 2, 3, 4, 5]
            if label in [0, 2, 5]:
                cls.selected_pred.append(pred[index])
                cls.selected_ref.append(ref[index])

    @classmethod
    def read_code2seq(cls, pred_label, result_dir):
        pred_path = os.path.join(result_dir, "pred.txt")
        ref_path = os.path.join(result_dir, "ref.txt")
        pred = cls._read_lines(pred_path)
        ref = cls._read_lines(ref_path)

        for index, label in enumerate(pred_label):
            # ['how_it_is_done' 'how_to_use' 'others' 'property' 'what' 'why']
            # [0, 1, 2, 3, 4, 5]
            if label in [1]:
                cls.selected_pred.append(pred[index])
                cls.selected_ref.append(ref[index])

    @classmethod
    def merge_and_write(cls):
        MERGE_RESULT = "/home/qiuyuanchen/Onedrive/my_parser/src/main/resources/merge_result"
        pred_merge = os.path.join(MERGE_RESULT, "pred.txt")
        ref_merge = os.path.join(MERGE_RESULT, "ref.txt")

        with open(pred_merge, "w") as f:
            for line in cls.selected_pred:
                f.write(line)
                f.write("\n")

        with open(ref_merge, "w") as f:
            for line in cls.selected_ref:
                f.write(line)
                f.write("\n")

    @staticmethod
    def _read_lines(path) -> list:
        lines = []
        with open(path) as f:
            for line in f.readlines():
                lines.append(line.strip())
        return lines


def evaluate_merge(validation_code, validation_label, testing_code, clf='rf'):
    clf = What2WriteClassifier()
    # clf.evaluate_classifier()
    pred_label_20000 = clf.build_classifier_and_predict(
        validation_code, validation_label, testing_code)
    
    DEEPCOM_RESULT = "/home/qiuyuanchen/Onedrive/EMSE-DeepCom/my_test"
    NNGEN_RESULT = "/home/qiuyuanchen/Onedrive/nngen/my_test"
    CODE2SEQ_RESULT = "/home/qiuyuanchen/Onedrive/code2seq-master/my_test"
    CompositeCodeSum.read_deepcom(pred_label_20000, DEEPCOM_RESULT)
    CompositeCodeSum.read_nngen(pred_label_20000, NNGEN_RESULT)
    CompositeCodeSum.read_code2seq(pred_label_20000, CODE2SEQ_RESULT)

    CompositeCodeSum.merge_and_write()


def main():
    clf = What2WriteClassifier()
    clf.evaluate_classifier('rf')
    clf.evaluate_classifier("naive")
    clf.evaluate_classifier("dt")
    clf.evaluate_classifier("lgbm")



if __name__ == "__main__":
    args = docopt(__doc__, version="CCCPM version 2.1")
    if args.get('clf'):
        print('clf')
        print(args.get('--classifier'))
        classifier = args.get('--classifier')
        clf = What2WriteClassifier()
        if classifier == "rf":
            clf.evaluate_classifier('rf')
        if classifier == "naive":
            clf.evaluate_classifier('naive')
        if classifier == "dt":
            clf.evaluate_classifier('dt')
        if classifier == "lgbm":
            clf.evaluate_classifier('lgbm')
        if classifier == "NN":
            print("cnn classifier")
            
        
    if args.get('eval'):
        clf = args.get('--classifier')
        if clf == "rf":
            evaluate_merge(EMPIRICAL_CODE, EMPIRICAL_LABEL, TEST_CODE, 'rf')
        if clf == "naive":
            evaluate_merge(EMPIRICAL_CODE, EMPIRICAL_LABEL, TEST_CODE, 'naive')
        if clf == "dt":
            evaluate_merge(EMPIRICAL_CODE, EMPIRICAL_LABEL, TEST_CODE, 'dt')
        if clf == "lgbm":
            evaluate_merge(EMPIRICAL_CODE, EMPIRICAL_LABEL, TEST_CODE, 'lgbm')
        