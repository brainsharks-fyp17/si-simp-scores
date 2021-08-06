from os import listdir
from os.path import isfile, join, isdir

all_data_dir = '/home/rumesh/Downloads/FYP/datasets/15m-tokenized/'


# create a new file
def getAllFilesRecursive(root):
    files = [join(root, f) for f in listdir(root) if isfile(join(root, f))]
    dirs = [d for d in listdir(root) if isdir(join(root, d))]
    for d in dirs:
        files_in_d = getAllFilesRecursive(join(root, d))
        if files_in_d:
            for f in files_in_d:
                files.append(join(root, f))
    return files


lst_file = getAllFilesRecursive(all_data_dir)
all_data = []
count = 0
for i in range(len(lst_file)):
    f = open(lst_file[i])
    data = f.readlines()
    for line in data:
        count += 1
        all_data.append(line.strip())
    f.close()
print("sentences read:", count)

cmplx = []
smpl = []
cmplx_limit = 10000000
smpl_limit = 1000000
cmplx_file = open("complex.si", "w")
simpl_file = open("simple.si", "w")
for line in all_data:
    words = line.split()
    if len(words) > 15:
        if cmplx_limit >= len(cmplx):
            cmplx.append(line)
            cmplx_file.write(line.strip() + "\n")
    elif len(words) > 5:
        if smpl_limit >= len(smpl):
            smpl.append(line)
            simpl_file.write(line.strip() + "\n")
print("complex:", len(cmplx))
print("simple:", len(smpl))
cmplx_file.close()
simpl_file.close()
