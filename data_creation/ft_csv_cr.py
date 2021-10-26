import pandas as pd
from sklearn.model_selection import train_test_split

simp_file = '../OCR-pdf/gov_comp_scarp_scl.txt'
cmplx_file = '../OCR-pdf/gov_comp_scarp_all.txt'

cmplex = [(line.strip(), 1) for line in open(cmplx_file).readlines()]
simple = [(line.strip(), 0) for line in open(simp_file).readlines()]
data = []
data.extend(cmplex)
data.extend(simple)
df = pd.DataFrame(data=data, columns=['sentence', 'class'])
df.to_csv('all_sup.csv', encoding='utf-8', columns=['sentence', 'class'], index=False, quoting=False)
train, test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
train = pd.DataFrame(data=train,columns=['sentence', 'class'])
test = pd.DataFrame(data=test,columns=['sentence', 'class'])
train.to_csv('train_sup.csv', encoding='utf-8', columns=['sentence', 'class'], index=False, quoting=False)
test.to_csv('test_sup.csv', encoding='utf-8', columns=['sentence', 'class'], index=False, quoting=False)
print(test[:3])
