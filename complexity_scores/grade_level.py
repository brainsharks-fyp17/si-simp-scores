import re
from os import listdir
from os.path import isfile, join, isdir
from typing import List, IO
from readability import is_sinhala_word, is_sinhala_syllable

base_path = "/home/rumesh/Documents/gov/"


def getAllFilesRecursive(root):
    files = [join(root, f) for f in listdir(root) if isfile(join(root, f))]
    dirs = [d for d in listdir(root) if isdir(join(root, d))]
    for d in dirs:
        files_in_d = getAllFilesRecursive(join(root, d))
        if files_in_d:
            for f in files_in_d:
                files.append(join(root, f))
    return files


grades = [2, 3, 4, 5, 6, 7, 9, "storybooks"]


def ari_stats(sentences: List[str]):
    n_sentences = len(sentences)
    n_words = 0
    n_chars = 0
    for sent in sentences:
        words = sent.split()
        for word in words:
            if is_sinhala_word(word):
                n_words += 1
                n_chars += len(word)
    return n_chars, n_words, n_sentences


def cal_ari(sentences: List[str]) -> float:
    a = 4.71
    b = 0.5
    c = -21.43
    n_chars, n_words, n_sentences = ari_stats(sentences)
    ari = a * (n_chars / n_words) + b * (n_words / n_sentences) + c
    ari = round(ari, 3)
    return ari


def cal_fre(sentences: List[str]) -> float:
    a = 206.835
    b = -1.015
    c = -84.6
    n_sent = len(sentences)
    n_words = 0
    n_syllables = 0
    for sent in sentences:
        for word in sent.split():
            if is_sinhala_word(word):
                n_words += 1
                for char in list(word):
                    if is_sinhala_syllable(char):
                        n_syllables += 1
    fre = a + b * (n_words / n_sent) + c * (n_syllables / n_words)
    fre = round(fre, 3)
    return fre


def cal_fkgl(sentences: List[str]) -> float:
    a = 0.39
    b = 11.8
    c = -15.59
    n_sent = len(sentences)
    n_words = 0
    n_syllables = 0
    for sent in sentences:
        for word in sent.split():
            if is_sinhala_word(word):
                n_words += 1
                for char in list(word):
                    if is_sinhala_syllable(char):
                        n_syllables += 1
    fkgl = a * (n_words / n_sent) + b * (n_syllables / n_words) + c
    fkgl = round(fkgl, 3)
    return fkgl


def do_grades():
    print("======ARI==========")
    for grade in grades:
        files = getAllFilesRecursive('/home/rumesh/Documents/scraped_txt')
        # files = getAllFilesRecursive(base_path + str(grade))
        # files = getAllFilesRecursive('/home/rumesh/Downloads/FYP/datasets/simpa-master')[:]
        lines = []
        for file in files:
            data = [line.strip() for line in open(file).readlines() if not line.isspace()]

            lines.extend(data)
        # ari_sc = cal_ari(lines)
        # fre_sc = cal_fre(lines)
        # fkgl_sc = cal_fkgl(lines)
        n_chars, n_words, n_sentences = ari_stats(lines)
        print((n_chars / n_words), (n_words / n_sentences), grade)
        # print("Grade", grade, ":", ari_sc)


def do_15m():
    print("======ARI==========")
    bag = 10
    simp = open("8.txt", "w")
    comp = open("11.txt", "w")
    # file_name = "/home/rumesh/Downloads/FYP/datasets/15m-tokenized/tokenized_shard_200000.txt"
    files = getAllFilesRecursive("/home/rumesh/Downloads/FYP/datasets/15m-tokenized/")
    for file_name in files:
        data = open(file_name).readlines()
        last = 0
        for i in range(bag, len(data), bag):
            sent = data[last:i]
            last = i
            val = cal_ari(sent)
            if val < 10:
                for j in sent:
                    simp.write(j)
            elif val > 11:
                for j in sent:
                    comp.write(j)


if __name__ == '__main__':
    cmp1 = "/home/rumesh/IdeaProjects/python/OCR-pdf/gov_comp_scarp_all.txt"
    cmp2 = "11.txt"
    # make comp 1M
    out = open("comp_train.txt", "w")
    all_sent = [line for line in open(cmp1).readlines()]
    all_sent.extend([line for line in open(cmp2).readlines()])
    for count, line in enumerate(all_sent):
        if count == 1000000:
            break
        if len(line.split()) > 5:
            out.write(line)
    out.close()
