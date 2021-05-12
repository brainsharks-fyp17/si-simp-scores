from cal_scores import word_length
from filter_cmplx_sent import create_output_file

dataset_file = 'all_words.txt'
if __name__ == '__main__':
    all_set = set()
    file = open(dataset_file)
    data = file.readlines()
    for line in data:
        line = line.strip()
        if word_length(line) < 11:
            all_set.add(line)
    lst = list(all_set)
    create_output_file(output_data=lst, ending="words_filtered")
