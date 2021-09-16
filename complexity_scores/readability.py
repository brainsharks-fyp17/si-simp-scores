from typing import List

short_vowels = [0xd85, 0xd87, 0xd89, 0xd8b, 0xd91, 0xd94]  # අ,ඇ,ඉ,....
long_vowels = [0xd86, 0xd88, 0xd8a, 0xd8c, 0xd92, 0xd95]  # ආ,ඈ,ඊ,...
pre_nas_cons = [0xd9f, 0xdac, 0xd9f, 0xd9f]  # ඟ,ඬ,ඳ,ඹ
voc_dia = [0xd93, 0xddb, 0xd96, 0xdde, 0xd8d, 0xdd8, 0xd8e, 0xdf2, 0xd8f, 0xddf, 0xd90, 0xdf3]  # ඓ,ෛ,ඖ,ෞ,...
ex_mis_pos = [0xd9b, 0xd9d, 0xda8, 0xdaa, 0xdae, 0xdb0, 0xdb5, 0xdb7]  # mahaprana ex: ඛ,ඨ,ඝ
mis_graph = [0xdc1, 0xdc2, 0xda1, 0xda3, 0xda4, 0xda3, 0xda4, 0xda5, 0xd9e, 0xdc6, 0xda6]  # ශ,ෂ,ඡ,ඣ,...
na_c = [0xdab]  # ණ
la_c = [0xdc5]  # ළ
zwj = [0x200d]  # zero-width joiner, ie. consonant conjunction

pillam_start = 0xdca
pillam_end = 0xdf4
vowel_start = sinhala_start = 0xd80
cons_start = vowel_end = 0xd9a
cons_end = 0xdc6
sinhala_end = 0xdf5

vowels = range(vowel_start, vowel_end)
cons = range(cons_start, cons_end)
pillam = range(pillam_start, pillam_end)
stop_words = [line.strip() for line in open('stop words.txt').readlines() if not line.isspace()]


def is_sinhala_syllable(letter):
    return vowel_start < ord(letter) < sinhala_end


def is_sinhala_letter(letter):
    unicode_val = ord(letter)
    # only Sinhala vowels and constants
    if sinhala_start <= unicode_val <= cons_end:
        return True


def is_sinhala_word(word):
    # A word has more than one Sinhala letter
    # can have other letters/digits; Ex: 25වන
    letter_count = 0
    for letter in word:
        if is_sinhala_letter(letter):
            letter_count += 1
            if letter_count >= 2:
                return True
    return False


def sent_stats(sentences: List[str]) -> dict:
    sent_count = len(sentences)
    word_count = 0
    stop_word_count = 0
    word_lengths = []
    for sent in sentences:
        words = sent.split()
        for word in words:
            if is_sinhala_word(word):
                word_count += 1
                word_lengths.append(len(word))
                if word in stop_words:
                    stop_word_count += 1

    return {
        'sent_count': sent_count,
        'word_count': word_count,
        'stop_word_count': stop_word_count,
        'stop_word_in_sent_per': round(stop_word_count * 100 / sent_count, 2),
        'stop_to_word_per': round(stop_word_count * 100 / word_count, 2),
        'avg_word_len': round(sum(word_lengths) / len(word_lengths), 2),
        'words_per_sent': round(word_count / sent_count, 2)
    }


def char_stats(sentences: List[str]) -> dict:
    out = dict()
    out['sentences'] = len(sentences)
    out['words'] = sum([len(sent.split()) for sent in sentences])
    out['chars'] = out['vowels'] = out['cons'] = out['pillam'] = out['short_vowels'] = out['long_vowels'] = out[
        'pre_nas_cons'] = out['voc_dia'] = out['voc_dia'] = out['ex_mis_pos'] = out['mis_graph'] = \
        out['na_c'] = out['la_c'] = out['zwj'] = 0
    for sent in sentences:
        for letter in sent:
            if sinhala_start < ord(letter) < sinhala_end:
                out['chars'] += 1
            if ord(letter) in vowels:
                out['vowels'] += 1
            if ord(letter) in cons:
                out['cons'] += 1
            if ord(letter) in pillam:
                out['pillam'] += 1
            if ord(letter) in short_vowels:
                out['short_vowels'] += 1
            if ord(letter) in long_vowels:
                out['long_vowels'] += 1
            if ord(letter) in pre_nas_cons:
                out['pre_nas_cons'] += 1
            if ord(letter) in voc_dia:
                out['voc_dia'] += 1
            if ord(letter) in ex_mis_pos:
                out['ex_mis_pos'] += 1
            if ord(letter) in mis_graph:
                out['mis_graph'] += 1
            if ord(letter) in na_c:
                out['na_c'] += 1
            if ord(letter) in la_c:
                out['la_c'] += 1
            if ord(letter) in zwj:
                out['zwj'] += 1
    return out


def get_avg(dct: dict, total: int) -> dict:
    out = dict()
    for string in dct.keys():
        out[string] = round(dct[string] * 100 / total, 2)
    return out


if __name__ == '__main__':
    simp_file = '../OCR-pdf/gov_comp_scarp_scl.txt'
    simp_lines = [i.strip() for i in open(simp_file).readlines()]
    st1 = char_stats(simp_lines)
    print("simple", "sentences:", st1['sentences'], get_avg(st1, st1['chars']))
    print(sent_stats(simp_lines))
    print("==================\n")

    cmplx_file = '../OCR-pdf/gov_comp_scarp_all.txt'
    cmplx_lines = [i.strip() for i in open(cmplx_file).readlines()]
    st = char_stats(cmplx_lines)
    print("complex", "sentences:", st['sentences'], get_avg(st, st['chars']))
    print(sent_stats(cmplx_lines))
