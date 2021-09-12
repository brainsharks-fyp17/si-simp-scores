from mtranslate import translate

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
file = open("ppdb-si-pivot.txt", "w")
j = 0
for word in source:
    j += 1
    try:
        paraphrases = set()
        for p in pivot_lang:
            pivot = translate(word, to_language=p, from_language=sinhala)
            out = translate(pivot, to_language=sinhala, from_language=p)
            paraphrases.add(out)
        print(j)
        if len(paraphrases) == 1 and list(paraphrases)[0] == word:
            continue
        file.write(word + ": ")
        for i in paraphrases:
            file.write(i + ",")
        file.write("\n")
        file.flush()
    except:
        pass
