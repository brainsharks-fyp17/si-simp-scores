from mtranslate import translate
import json

normal_file_path = '../datasets/test.8turkers.tok.norm'
simple_file_path = '../datasets/test.8turkers.tok.simp'
output_file_path = '../output/output_turk.txt'
i = 0
complex_file = open(normal_file_path)
simple_file = open(simple_file_path)
output_file = open(output_file_path, mode='a', encoding='utf8')


def read_pair():
    global i
    cl = complex_file.readline().strip().replace("-RRB-","").replace("-LRB-","")
    sl = simple_file.readline().strip().replace("-RRB-","").replace("-LRB-","")
    i = i + 1
    return i, cl, sl


def translate_pair(cmplx, simple):
    nt = translate(cmplx, 'si').replace('\u200d', '')
    st = translate(simple, 'si').replace('\u200d', '')
    return nt, st


def convert_data(idx, ce, se, cs, ss):
    data = dict()
    data["i"] = idx
    data["CEn"] = ce
    data["SEn"] = se
    data["CSi"] = cs
    data["SSi"] = ss
    return json.dumps(data, ensure_ascii=False)


def write_data(json_val):
    output_file.write(json_val + ",\n")
    output_file.flush()


for i in range(10):
    j, cen, sen = read_pair()
    csi, ssi = translate_pair(cen, sen)
    json_val = convert_data(j, cen, sen, csi, ssi)
    print(json_val)
    write_data(json_val)
output_file.close()
