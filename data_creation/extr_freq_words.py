simp_file = open('simple.si').readlines()
complex_file = open('complex.si').readlines()
stop_file = open('stop.list').readlines()
stop_words = [word.strip() for word in stop_file]


def count_tokens(lines):
    word_dict = dict()
    for line in lines:
        line = line.strip().split()
        for token in line:
            if token not in stop_words:
                if token not in word_dict:
                    word_dict[token] = 1
                else:
                    word_dict[token] += 1
    print("counted!")
    return word_dict


if __name__ == '__main__':
    import json

    c = count_tokens(complex_file)
    s = count_tokens(simp_file)
    json.dump(c, ensure_ascii=False, fp=open("comp_words.json", "w"), indent=4)
    json.dump(s, ensure_ascii=False, fp=open("simp_words.json", "w"), indent=4)
