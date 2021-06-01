from cal_scores import is_strictly_sinhala_word, word_length
from filter_cmplx_sent import create_output_file

dataset_file_path = '/home/rumesh/Downloads/FYP/datasets/common-crawl-si.txt'
datset2 = '../datasets/parallel-08.11.2020-Tr74K.si-en.si'

if __name__ == '__main__':
    all_words_set = set()
    file = open(datset2)
    file2 =open(dataset_file_path)
    sentences = file.readlines()
    sentences.extend(file2.readlines())
    file2.close()
    file.close()
    for sent in sentences:
        words = sent.split(" ")
        for word in words:
            if is_strictly_sinhala_word(word):
                all_words_set.add(word)
    all_words_list = list(all_words_set)
    create_output_file(output_data=all_words_list, ending="all_words", extension="txt")
