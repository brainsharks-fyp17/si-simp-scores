import codecs
import datetime
import json

from cal_scores import words_in_sentence, dataset_file_path, is_sinhala_word, word_length

MAX_NO_PHRASES_IN_SENT = 2
SENTENCE_LEN_THRESHOLD = 30
SPW_THRESH = 5
SPW_IN_SENT_THRESH = 5


# cluttered sentences contain phrases connected by commas
def is_cluttered_sent(sent):
    return sent.count(",") >= MAX_NO_PHRASES_IN_SENT or sent.count(" . ") >= MAX_NO_PHRASES_IN_SENT


def filter_long_sentences(sent_list, threshold):
    result = []
    for sent in sent_list:
        sent = sent.strip()
        if words_in_sentence(sent) > threshold and (not is_cluttered_sent(sent)):
            result.append(sent)
    return result


def filter_by_syllables_per_word(sent_list, word_len_thresh, freq_thresh):
    result = []
    for sent in sent_list:
        words = sent.split(" ")
        freq = 0
        for word in words:
            if is_sinhala_word(word) and word_length(word) >= word_len_thresh:
                freq += 1
        if freq >= freq_thresh:
            result.append(sent)
    return result


def create_output_file(output_data, ending="", extension="txt"):
    # creates a unique file name at each run
    s = datetime.datetime.now().strftime("%d-%B-%Y-%I-%M-%S-%f-%p") + "_" + ending + "." + extension
    with open(s, "w", encoding='utf8') as f:
        if type(output_data) == list:
            string_data = ""
            for line in output_data:
                string_data += line + "\n"
            f.write(string_data)
        elif type(output_data) == str:
            f.write(output_data)
        elif type(output_data) == dict:
            json.dump(output_data, f, ensure_ascii=False)
        else:
            raise TypeError("Unsupported input type")


def for_sl_db():
    file_data = open(dataset_file_path)
    data = file_data.readlines()
    print("Number of all sentences: ", len(data))
    filtered_long = filter_long_sentences(data, SENTENCE_LEN_THRESHOLD)
    print("Number of long sentences: ", len(filtered_long))
    create_output_file(filtered_long, "Len_" + str(len(filtered_long)), "txt")

    filtered_spw_sent = filter_by_syllables_per_word(filtered_long, SPW_THRESH, SPW_IN_SENT_THRESH)
    print("Filtered by SPW: ", len(filtered_spw_sent))
    create_output_file(filtered_spw_sent, "SPW_" + str(len(filtered_spw_sent)), "txt")


if __name__ == '__main__':
    file_data = open(dataset_file_path, encoding='utf8')
    data = file_data.readlines()
    print("Number of all sentences: ", len(data))
    filtered_long = filter_long_sentences(data, SENTENCE_LEN_THRESHOLD)
    print("Number of long sentences: ", len(filtered_long))
    # create_output_file(filtered_long, "Len_"+str(len(filtered_long)), "txt")
    #
    filtered_spw_sent = filter_by_syllables_per_word(filtered_long, SPW_THRESH, SPW_IN_SENT_THRESH)
    print("Filtered by SPW: ", len(filtered_spw_sent))
    create_output_file(filtered_spw_sent, "CommonCrawl_SPW_" + str(len(filtered_spw_sent)), "txt")
