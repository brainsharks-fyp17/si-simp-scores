from os import listdir
from os.path import isfile, join, isdir
import pdf2image

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract


def getAllFilesRecursive(root):
    files = [join(root, f) for f in listdir(root) if isfile(join(root, f))]
    dirs = [d for d in listdir(root) if isdir(join(root, d))]
    for d in dirs:
        files_in_d = getAllFilesRecursive(join(root, d))
        if files_in_d:
            for f in files_in_d:
                files.append(join(root, f))
    return files


def pdf_to_img(pdf_file):
    return pdf2image.convert_from_path(pdf_file)


def ocr_core(file):
    text = pytesseract.image_to_string(file, lang="sin")
    return text


def print_pages(pdf_file):
    out = []
    images = pdf_to_img(pdf_file)
    for pg, img in enumerate(images):
        out.append(str(ocr_core(img)))
    return out


if __name__ == '__main__':
    pdf_dir = r"/home/rumesh/Documents/pdftoimage/sinhala la  G-7"
    files = getAllFilesRecursive(pdf_dir)
    for file_name in files:
        out_file = open(file_name + ".txt", "w")
        txt = print_pages(file_name)
        for i in txt:
            out_file.write(i + "\n")
        out_file.close()
