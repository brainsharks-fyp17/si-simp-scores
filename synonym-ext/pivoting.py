import multiprocessing
import uuid

english = "en"
sinhala = "si"
german = "de"
chinese = "zh-CN"
arabic = "ar"
french = "fr"
russian = "ru"
turkish = "tr"
thai = "th"
pivot_lang = [german, chinese, arabic, french, russian, turkish, thai]

f = open("all_words.txt").readlines()
source = [line.strip() for line in f]


def do_pivoting(start, end):
    uid = str(uuid.uuid4())
    file = open("ppdb-si-pivot-" + uid + ".txt", "w")
    from mtranslate import translate
    print("started", start, end)
    for k in range(start, end):
        try:
            paraphrases = set()
            for p in pivot_lang:
                pivot = translate(source[k], to_language=p, from_language=sinhala)
                out = translate(pivot, to_language=sinhala, from_language=p)
                paraphrases.add(out)
            if len(paraphrases) == 1 and list(paraphrases)[0] == source[k]:
                continue
            file.write(source[k] + ": ")
            for i in paraphrases:
                file.write(i + ",")
            file.write("\n")
            file.flush()
        except:
            pass
    print("ended", start, end)
    file.close()


n_processes = 5
N = 10
pool = multiprocessing.Pool(n_processes)
chunks = [(i, i + N) for i in range(0, len(source), N)]
print(chunks)
pool.starmap(do_pivoting, chunks)
