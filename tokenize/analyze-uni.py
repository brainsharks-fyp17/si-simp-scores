sinhala_start = 3456
vowels_and_const_end = 3527
sinhala_end = 3573

input_file = '/home/rumesh/Downloads/FYP/datasets/15m-tokenized/tokenized_shard_200000.txt'


def is_sinhala_letter(letter):
    unicode_val = ord(letter)
    # only Sinhala vowels and constants
    if sinhala_start <= unicode_val <= vowels_and_const_end:
        return True


sent = "විද්‍යාව"  #
a = " රචිත බුද්ධපුත්‍ර හිමියන් රාත්‍රී අහස වර්ණනා කර ඇත්තේ මේ ආකාරයට ය.,කනේ ශල්‍යකර්මයක් සඳහා රෝහල් ගත කර " \
    "සිටි මෙම 12 " \
    "හැවිරිදි පිරිමි දරුවා හලාවත මයික්කුලම අගමැතිවරයා සම්බන්ධ දෙවැනි පරීක්‍ෂණයකට එළැඹීමට මෙය හොඳ අවස්ථාවක් වනු " \
    "ඇතැයි ද පවතින වාතාවරණය "
# sent = sent.split(" ")
print(len(sent))
s = list(sent)
se = " ".join(s)

chrs = [str(ord(i)) for i in s]
chrs_sent = " ".join(chrs)
fl = open("out.txt", "w")
fl.write(se + "\n\n")
fl.write(chrs_sent)
sent_uni = []
