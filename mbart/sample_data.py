import random
from os import listdir
from os.path import isdir, isfile, join

start = 51
end = 1001
lst = list(range(start, end))
choices = random.sample(lst, 50)
choices.sort()
# print(choices)

# got from a run
choices = [51, 53, 66, 72, 96, 118, 121, 189, 263, 276, 277, 278, 321, 356, 360, 394, 417, 433, 445, 462, 472, 475,
           501, 512, 554, 557, 597, 659, 716, 724, 728, 737, 752, 758, 776, 783, 817, 819, 832, 849, 850,
           852, 876, 880, 890, 917, 919, 960, 968, 978]
assert len(choices) == 50


def getAllFilesRecursive(root):
    files = [join(root, f) for f in listdir(root) if isfile(join(root, f))]
    dirs = [d for d in listdir(root) if isdir(join(root, d))]
    for d in dirs:
        files_in_d = getAllFilesRecursive(join(root, d))
        if files_in_d:
            for f in files_in_d:
                files.append(join(root, f))
    return files


directory = "results"
files = getAllFilesRecursive(directory)
print("Files found: " + str(len(files)))

for file in files:
    file_name = file.split("/")[1]
    selected = []
    lines = open(file).readlines()
    for i in range(len(lines)):
        if i in choices:
            selected.append(lines[i])
    out_file = open("selected/" + file_name, "w")
    for line in selected:
        out_file.write(line.strip()+"\n")
    out_file.close()