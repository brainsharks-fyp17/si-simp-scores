normal_file_path = '../datasets/ls.original'
output_file_path = '../output/ls.without-duplicates.original'
i = 0
complex_file = open(normal_file_path)
output_file = open(output_file_path, mode='a', encoding='utf8')
for line in complex_file:
    ln = line.strip()
    if i % 3 == 0:
        output_file.write(ln+"\n")
        output_file.flush()
    i+=1
output_file.close()
