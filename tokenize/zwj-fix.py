import re
from typing import Text

# unicode values of chars
zwj = 8205  # \u200d
halkirima = 3530  # '්'
rayanna = 3515  # 'ර',
yayanna = 3514  # 'ය'

sinhala_start = 3456
vowels_and_const_end = 3527
sinhala_end = 3573


def is_sinhala_letter_or_allowed_sym(letter):
    if letter == " " or letter == "#":  # or letter == "$":
        return True
    unicode_val = ord(letter)
    # only Sinhala vowels and constants
    if sinhala_start <= unicode_val <= sinhala_end:
        return True


def fix_zwj(text: Text) -> Text:
    """
    Fixes Zero Width Joiner issue in Sinhala Language data
    Will break any other language/ symbol that relies on Zero Width Joiner
    :param text: Input sentence/word
    :return: Fixed sentence/ word
    """
    chr_list = [ord(c) for c in list(text) if ord(c) != 8205]  # convert to unicode value, remove zero width joiners
    new_chr_list = []
    ln_chr_list = len(chr_list)
    for i in range(ln_chr_list):
        new_chr_list.append(chr_list[i])
        if i < ln_chr_list - 1:
            if chr_list[i] == halkirima and (chr_list[i + 1] == rayanna or chr_list[i + 1] == yayanna):
                new_chr_list.append(zwj)  # add zwj
    new_chr_list = [chr(uni) for uni in new_chr_list]
    return "".join(new_chr_list)


def clean(text: Text) -> Text:
    """
    Remove all non Sinhala characters.
    Replace numbers with '#' and English words with '$'
    :param text: Sinhala sentence
    :return: cleaned Sinhala sentence
    """
    text = text.strip()
    text = re.sub(r"[0-9]+", "#", text)
    # text = re.sub(r"\b[A-Za-z]+", "$", text)

    cleaned_sent = ''
    for char in text:
        if is_sinhala_letter_or_allowed_sym(char):
            cleaned_sent += char
    tokens = cleaned_sent.split()  # split on whitespaces
    sent = ' '.join(tokens)
    return sent


if __name__ == '__main__':
    sentences = ["ඉන්දුනීසියාවේ සුමාත්‍රා දිවයිනේ මුසී ගංඟාවේ මගී යාත්‍රාවක් අනතුරට ලක්වීමෙන් පුද්ගලයන් 13 දෙනෙකු ",
                 "ජීවිතක්ෂයට පත්වුණා Hello , 6 39 // * chi sk s erefකුසලානෙට සි‍ලෝන්vfrv  බාබේරfvcfියන්",
                 "බ්‍රැඩ්බි කුසලානෙට ට්‍රිනිට් එකත් එක්ක සෙල්ලම් කරපු එකයි සි‍ලෝන් බාබේරියන්",
                 "බ්රැඩ්බි කුසලානෙට ට්රිනිට් එකත් එක්ක ",
                 "බෞද්ධ කටයුතු දෙපාර්තමේන්තුව",
                 "ශාස්ත්රීය භාවිතයක් විවේචනය ",
                 "ශාස්ත්‍රීය භාවිතයක් විවේචනය ",
                 "මාධ්‍ය සදාචාරය යහපත් ක්රියාකාරිත්වය",
                 "එවැනි තක්සේරුවක් තමයි ඔවුන් පේ‍්‍රක්ෂකයින්ට ලබා දීලා තියෙන්නෙ",
                 "අධි වේගී අධි ධාරිතා ‍ෆ්‍රී වයි ෆයි සේවය ලබා දුන්න",
                 "ප්‍රසාරණය වීම සඳහා මේ ව්‍යාපාරවලට ප්‍රාග්ධනය අවශ්‍යයි ",
                 "දරුණු ප‍්‍රහාරයක් එල්ල කළද ඒවා ව්‍යර්ථ",
                 "අග්‍රාමාත්‍ය රනිල් වික්‍ර‍මසිංහ"
                 ]
    r = []
    for i in sentences:
        r.append((clean(i)))
    open("out-zwj.txt", "w").write('\n'.join(r))
