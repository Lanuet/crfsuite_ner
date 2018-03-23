import codecs

import numpy as np
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

def read_conll_file(path):
    sentences = []
    sentence = []

    lines = []
    with codecs.open(path, 'r', 'utf-8') as file:
        for line in file:
            if line.strip() == "":
                sentence = tuple(sentence)
                sentences.append(sentence)
                sentence = []
            else:
                tokens = line.strip().split()
                for token in tokens:
                    lines.append(token)
                lines = tuple(lines)
                sentence.append(lines)
                lines = []
    return sentences

def word2features(sent, i):
    if (len(sent) < 3):
        print("fail")
    word = sent[i][0]
    postag = sent[i][1]
    chunktag = sent[i][2]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'chunktag' : chunktag,
        'chunktag[:2]' : chunktag[:2]
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        chunktag1 = sent[i-1][2]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:chunktag': chunktag1,
            '-1:chunktag[:2]': chunktag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        chunktag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:chunktag': chunktag1,
            '+1:chunktag[:2]': chunktag1[:2],

        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, chunktag, label in sent]

def sent2tokens(sent):
    return [token for token, postag,chunktag, label in sent]

def main():
    train_sents = read_conll_file("ner/ner_train.muc")
    test_sents = read_conll_file("ner/ner_test.muc")
    input_train = [sent2features(s) for s in train_sents]
    output_train = [sent2labels(s) for s in train_sents]
    input_test = [sent2features(s) for s in test_sents]
    output_test = [sent2labels(s) for s in test_sents]
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=20,
        all_possible_transitions=False,
    )
    crf.fit(input_train, output_train)
    labels = list(crf.classes_)
    y_pred = crf.predict(input_test)
    metrics.flat_f1_score(output_test, y_pred,
                          average='weighted', labels=labels)
    # bảng tổng hợp theo nhãn
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    with open("output.txt", "w",encoding='utf-8') as f:
        f.write(metrics.flat_classification_report(
        output_test, y_pred, labels=sorted_labels, digits=3
    ))

    print(metrics.flat_classification_report(
        output_test, y_pred, labels=sorted_labels, digits=3
    ))

if __name__ == '__main__':
    main()