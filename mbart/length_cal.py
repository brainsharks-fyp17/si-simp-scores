from os import listdir
from os.path import isdir, isfile, join


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

for file_name in files:
    file_data = [i.strip() for i in open(file_name).readlines()]
    lines = len(file_data)
    words = 0
    for line in file_data:
        words += len(line.split())  # simply split using spaces
    ratio = round((words / lines), 2)
    print(file_name.split("/")[-1],ratio)
