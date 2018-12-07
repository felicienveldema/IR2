import pickle
import numpy as np
import string
import re

from nltk.corpus import stopwords
from collections import OrderedDict, defaultdict

TARGET_COUNT = 2

def read_txt(src, pickle_src):
    with open(pickle_src, "rb") as f:
        pmi = pickle.load(f)
        pmi = {k: OrderedDict(sorted(v.items(), key=lambda x:x[1], reverse=True)) for k,v in pmi.items()}
    with open(src, "r") as f:
        data = f.readlines()
    return data, pmi

def modify_txt(data, pmi):
    stopwords_set = set(stopwords.words('english'))
    RETOK = re.compile(r'\w+|[^\w\s]|\n', re.UNICODE)
    for i, line in enumerate(data):
        line_split = line.split("\t")
        input_txt = line_split[0].split("text:")[1]
        label_txt = line_split[1]
        sentence_topic_words = []
        words = np.array([word for word in RETOK.findall(input_txt) if word != "__unk__" and word not in stopwords_set and word not in string.punctuation])
        for word in words:
            sentence_topic_words.extend(list(pmi[word].keys())[:2])
        new_label_txt = label_txt + "¡" + " ".join(sentence_topic_words)
        data[i] = "text:" + input_txt + "\t" + new_label_txt + "\t" + line_split[2]
    return data

def write_txt(data, src):
    with open(src, "w+") as f:
        f.writelines(data)
    return


def main():
    src = "data/Twitter/train.txt"
    new_src = "data/Twitter/train_small_modified.txt"
    pmi_src = "data/Twitter/pmi_nested.pkl"
    data, pmi = read_txt(src, pmi_src)
    data = modify_txt(data, pmi)
    write_txt(data, new_src)


if __name__ == '__main__':
    main()