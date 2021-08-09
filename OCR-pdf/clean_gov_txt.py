from typing import List, IO

file_names = ['pts g-7 S.pdf.txt']
short_line_threshold = 6  # chars


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


def write_to_file(lst: List[str], file_obj: IO) -> None:
    for i in lst:
        file_obj.write(i)
    file_obj.close()


def crate_output_name(name: str) -> str:
    name = file.split(".txt")
    name = name[:-1]
    name.append(".cleaned.txt")
    name = "".join(name)
    return name


for file in file_names:
    print(file)
    lines = open(file).readlines()
    out_name = crate_output_name(file)
    out_file = open(out_name, "w")
    lines = remove_short_lines(lines)
    lines = remove_copyright(lines)
    write_to_file(lines, out_file)
