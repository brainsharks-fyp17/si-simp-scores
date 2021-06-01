import json
from os import listdir
from os.path import isfile, join, isdir

out_file = "Raw-news-sinhala.txt"
failed_list = "failed.list"

def getAllFilesRecursive(root):
    files = [join(root, f) for f in listdir(root) if isfile(join(root, f))]
    dirs = [d for d in listdir(root) if isdir(join(root, d))]
    for d in dirs:
        files_in_d = getAllFilesRecursive(join(root, d))
        if files_in_d:
            for f in files_in_d:
                files.append(join(root, f))
    return files


if __name__ == '__main__':
    lst = getAllFilesRecursive("/home/rumesh/Downloads/Raw News Collection/Sinhala")
    # out = open(out_file,"w")
    failed_files = []
    for file in lst:
        try:
            with open(file) as f:
                data = json.load(f)
                cont = data['Content']
                # for line in cont:
                #     out.write(line+"n")
        except Exception:
            failed_files.append(file)
    # out.close()
    list_file = open(failed_list,"w")
    for i in failed_files:
        list_file.write(i+"\n")

