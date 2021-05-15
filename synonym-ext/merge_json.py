import json
import os

folder = '../datasets/morph_original'
onlyFiles = [os.path.join(folder, f) for f in os.listdir(folder) if
             os.path.isfile(os.path.join(folder, f))]
print(onlyFiles)

final_dict = dict()
for i in onlyFiles:
    with open(str(i)) as f:
        data = json.load(f)
        final_dict.update(data)

with open("si_neighbors_unfiltered.json", "w") as f:
    json.dump(final_dict, f, ensure_ascii=False)
