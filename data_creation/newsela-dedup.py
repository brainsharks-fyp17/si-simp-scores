comp = open("/home/rumesh/Downloads/FYP/datasets/newsela-sinhala/complex.txt").readlines()
simp = open("/home/rumesh/Downloads/FYP/datasets/newsela-sinhala/simple.txt").readlines()
dedup_comp = open("newsela-comp-si-dedup.txt", "w")
dedup_simp = open("newsela-simp-si-dedup.txt", "w")
assert len(comp) == len(simp)
cmp = dict()
smp = dict()
for i, line in enumerate(comp):
    if line not in cmp:
        cmp[line] = i

for i, line in enumerate(simp):
    if line not in smp:
        smp[line] = i

for line, i in cmp.items():
    for s_line in smp:
        if i == smp[s_line]:
            dedup_comp.write(line)
            dedup_simp.write(s_line)
dedup_simp.close()
dedup_comp.close()
