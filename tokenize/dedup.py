input_file = 'all_data_unfiltered_si.txt'
output_file = 'all_data_dedupe_si.txt'
if __name__ == '__main__':
    f = open(input_file)
    lines = f.readlines()
    lines_set = set()
    for i in lines:
        i = i.strip()
        if i != "" or i != "\n":
            lines_set.add(i)
    f.close()
    f_out = open(output_file, mode='w')
    for i in list(lines_set):
        i = i + "\n"
        f_out.write(i)
    f_out.close()
