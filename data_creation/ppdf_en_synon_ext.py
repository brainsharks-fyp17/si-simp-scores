from mtranslate import translate


def ppdb_trans():
    out_file = open("ppdb-si-raw.txt", "w")
    ppdb_data = [x.strip().replace("turkers=", "").replace("\"", "")[4:] for x in open("ppdb-en").readlines()]
    for line in ppdb_data:
        lst = list(set(line.split("\t")))
        for word in lst:
            out_file.write(str(translate(word, 'si')) + "\t")
        out_file.write("\n")
        out_file.flush()
    out_file.close()


def fre_trans():
    out_file = open("freq-en.txt","w")
    freq_lies = [x.strip() for x in open("frequent_comp.list").readlines()]
    for i in freq_lies:
        out_file.write(translate(i)+"\n")