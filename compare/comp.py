orig = 'complex-1000.txt'
# first = "sim"
comp = open(orig).readlines()
simp = open('simp1000-nimasha2.txt').readlines()
duplicates = []


def replaceMultiple(mainString, toBeReplaces, newString):
    for elem in toBeReplaces:
        if elem in mainString:
            mainString = mainString.replace(elem, newString)

    return mainString


def rep(s):
    s1 = replaceMultiple(s, ["…", "\\", '{', '}', '[', ']', '॥', '#', '”', '“', '.', ',', ';', ':', '/', '"', '–', '-',
                             '*',
                             '(', ')', '\'', '%', '$', '&', '+', '=', '<', '>', '|', '—', '_', '\ufeff', '\u200c', '’',
                             '‘','.'],
                         '')
    return s1


for i, line in enumerate(comp):
    w1 = list(set(rep(line).split()))
    w2 = list(set(rep(simp[i]).split()))
    l1 = len(w1)
    l2 = len(w2)
    # if w1 == w2:
    #     duplicates.append(i+1)
    exact = 0
    for j in w1:
        if j in w2:
            exact += 1
    if 0 < abs(exact - l1) < 3:
        duplicates.append(i + 1)
print(len(duplicates))
print(duplicates)
