from cal_scores import *


def filter_long_sentences(sent_list, threshold):
    result = []
    for sent in sent_list:
        sent = sent.strip().replace('\u200d', '')
        if words_in_sentence(sent) > threshold:
            result.append(sent)
    return result


def create_output_file(output_data, ending="", extension="txt"):
    s = str(uuid.uuid4()) + "_" + ending + "." + extension
    with open(s, "w") as f:
        if type(output_data) == list:
            string_data = ""
            for line in output_data:
                string_data += line + "\n"
            f.write(string_data)
        elif type(output_data) == str:
            f.write(output_data)
        else:
            raise TypeError("Un supported format")


if __name__ == '__main__':
    file_data = open(dataset_file_path)
    data = file_data.readlines()
    filtered = filter_long_sentences(data, 70)
    print("Number od sentences: ", len(filtered))
    print(filtered)
    create_output_file(filtered, "", "txt")
