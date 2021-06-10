sinhala_start = 3456
vowels_and_const_end = 3527
sinhala_end = 3573

input_file = '/home/rumesh/Downloads/FYP/datasets/15m-tokenized/tokenized_shard_200000.txt'
def is_sinhala_letter(letter):
    unicode_val = ord(letter)
    # only Sinhala vowels and constants
    if sinhala_start <= unicode_val <= vowels_and_const_end:
        return True

f = open(input_file)
lines = f.readlines()
words = []
for line in lines:
    lst = line.split()
    words.extend(lst)
chars = set()
for word in words:
    chars.update(set(list(word)))
chars = list(chars)
chars = sorted(chars)
print(chars)