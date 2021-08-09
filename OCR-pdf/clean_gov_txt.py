import re
from os import listdir
from os.path import isfile, join, isdir
from typing import List, IO

short_line_threshold = 6  # chars


def getAllFilesRecursive(root):
    files = [join(root, f) for f in listdir(root) if isfile(join(root, f))]
    dirs = [d for d in listdir(root) if isdir(join(root, d))]
    for d in dirs:
        files_in_d = getAllFilesRecursive(join(root, d))
        if files_in_d:
            for f in files_in_d:
                files.append(join(root, f))
    return files


def remove_short_lines(sentences: List[str]) -> List[str]:
    out = []
    for line in sentences:
        if line == "\n" or line == " \n":
            out.append(line)
        elif len(line) > short_line_threshold:
            out.append(line)
    return out


def remove_copyright(sentences: List[str]) -> List[str]:
    out = []
    for line in sentences:
        if ("නොමිලේ බෙදා හැරීම සඳහා ය" not in line) or ("සියලු හිමිකම්‌ ඇවිරිණි" not in line):
            out.append(line)
    return out


def remove_english(sentences: List[str]) -> List[str]:
    out = []
    patt = re.compile(r"\(*[0-9]*\)")
    for line in sentences:
        if re.match(patt, line):
            line = line.replace(r"\(*[0-9]*\)", "")
        out.append(line)
    return out


def write_to_file(lst: List[str], file_obj: IO) -> None:
    for i in lst:
        file_obj.write(i)
    file_obj.close()


def concat_and_split(sentences: List[str]) -> List[str]:
    sentences.append("\n")
    single_line = ""
    for i in range(len(sentences) - 1):
        if sentences[i + 1] == "\n":
            single_line += (sentences[i].replace("\n", " ") + ".")
        else:
            single_line += sentences[i].replace("\n", " ")
    single_line = single_line.replace("\n", " ").replace("~", "").replace("%", "").replace("*", "")
    out = single_line.split(".")
    return [line.strip() + "\n" for line in out if not line.isspace() and len(line) > 1]


def stat_remove(sentences: List[str]) -> List[str]:
    out = []
    for line in sentences:
        sp = line.split()
        if len(line) > 0 and len(sp) > 0 and len(line) / len(sp) < 4:
            # print(line)
            continue
        else:
            out.append(line)
    return out


def crate_output_name(name: str) -> str:
    name = file.split(".txt")
    name = name[:-1]
    name.append(".cleaned.txt")
    name = "".join(name)
    return name


file_names = getAllFilesRecursive("/home/rumesh/Documents/gov")
for file in file_names:
    print(file)
    lines = open(file).readlines()
    out_name = crate_output_name(file)
    out_file = open(out_name, "w")
    lines = remove_short_lines(lines)
    lines = stat_remove(lines)
    lines = remove_copyright(lines)
    lines = remove_english(lines)
    lines = concat_and_split(lines)
    lines = remove_short_lines(lines)
    lines = stat_remove(lines)
    write_to_file(lines, out_file)
