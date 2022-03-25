import time

from mtranslate import translate

english = "en"
sinhala = "si"
german = "de"
pivot_lang = [german, english]

lines = open("SiTa-56K-gt8.txt").readlines()
lines = [line.strip() for line in lines]

outfile = open("out.txt", "w")
print("Started")
for i in range(len(lines)):
    for p in pivot_lang:
        pivot = translate(lines[i], to_language=p, from_language=sinhala)
        out = translate(pivot, to_language=sinhala, from_language=p)
        outfile.write(lines[i] + "\t" + out + "\n")
        outfile.flush()
    print(i)
    if (i+1) % 1000 == 0:
        time.sleep(3600)

outfile.close()
