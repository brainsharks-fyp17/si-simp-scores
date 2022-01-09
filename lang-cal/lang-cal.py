en_start = 65
en_end = 122
si_start = 3456
si_end = 3573
file_name = "prediction-model8.2-mbart.txt"
lines = open(file_name).readlines()
si_word_count = 0
en_word_count = 0
other_word_count = 0
total_word_count = 0


def is_si_word(word, p=0.3):
    count = 0
    chr_list = list(word)
    for ch in chr_list:
        if si_start <= ord(ch) <= si_end:
            count += 1
    return count / len(chr_list) > p


def is_en_word(word, p=0.3):
    count = 0
    chr_list = list(word)
    for ch in chr_list:
        if en_start <= ord(ch) <= en_end:
            count += 1
    return count / len(chr_list) > p


for line in lines:
    words = line.strip().split()
    for word in words:
        total_word_count += 1
        if is_si_word(word):
            si_word_count += 1
        elif is_en_word(word):
            en_word_count += 1
        else:
            other_word_count += 1

print("Sinhala words:", si_word_count)
print("English words:", en_word_count)
print("Other words:", other_word_count)
print("Sinhala %:", si_word_count/(si_word_count+en_word_count)*100)
print("English %:", en_word_count/(si_word_count+en_word_count)*100)
# print("Other %:", other_word_count/total_word_count*100)
