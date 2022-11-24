import random

choices = [51, 53, 66, 72, 96, 118, 121, 189, 263, 276, 277, 278, 321, 356, 360, 394, 417, 433, 445, 462, 472, 475,
           501, 512, 554, 557, 597, 659, 716, 724, 728, 737, 752, 758, 776, 783, 817, 819, 832, 849, 850,
           852, 876, 880, 890, 917, 919, 960, 968, 978]
assert len(choices) == 50

file_data = [
    [i.strip() for i in open("human/1.txt").readlines()],
    [i.strip() for i in open("human/2.txt").readlines()],
    [i.strip() for i in open("human/3.txt").readlines()]
]

out_file = open("human_mix.txt", "w")
for i in choices:
    selected_file = random.choice(file_data)
    out = selected_file[i]
    out_file.write(out + "\n")
out_file.close()
