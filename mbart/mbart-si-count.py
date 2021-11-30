sinhala_start = 0xd80
sinhala_end = 0xdf5
file = open("dict.txt").readlines()
# file = open("dict.si_LK.txt").readlines()
print("Lines: ",len(file))
si_count = 0
si_lines = ""
for line in file:
    chars = list(line)
    for char in chars:
        if sinhala_start < ord(char) < sinhala_end :
            si_count += 1
            si_lines += ''.join(chars)
            break

print("Si lines: ",si_count)
si = open("si_mbart25_dict.txt","w")
si.write(si_lines)
