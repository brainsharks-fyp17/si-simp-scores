import progressbar

input_file = 'all_data_dedupe_si.txt'
out_file = 'all_data_dedupe_sort_si.txt'

sinhala_start = 3456
vowels_and_const_end = 3527
sinhala_end = 3573


def is_sinhala_letter_or_space(letter):
    if letter == " ":
        return True
    unicode_val = ord(letter)
    # only Sinhala vowels and constants
    if sinhala_start <= unicode_val <= vowels_and_const_end:
        return True


file = open(input_file)
lines = file.readlines()
filtered_lines = []
for i in range(len(lines)):
    line = lines[i]
    filtered_line = ""
    for char in line:
        if is_sinhala_letter_or_space(char):
            filtered_line += char
    filtered_line = filtered_line.strip()
    filtered_line += "\n"
    filtered_lines.append(filtered_line)

print()
file.close()
lines_sorted = sorted(filtered_lines)
out = open(out_file, "w")
for i in lines_sorted:
    out.write(i)
out.close()
