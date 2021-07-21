vocab_file = '/home/rumesh/Downloads/FYP/sentencepiece-vocabs/m-30000-bpe.vocab'
out_file = 'si-vocab-30000.list'
if __name__ == '__main__':
    with open(vocab_file) as f:
        lines = f.readlines()
        out = open(out_file, "w")
        for line in lines:
            line = line.split("\t")
            assert len(line) == 2
            word = line[0]
            if "▁" not in word:
                word = "##" + word
            else:
                word = word[1:]
            out.write(word + "\n")
        out.close()
