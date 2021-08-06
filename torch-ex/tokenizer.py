import collections
import unicodedata
import os
import sentencepiece as spm


txt = """
ඩෙංගු රෝගීන්ගෙන් ආසන්න වශයෙන් 423 ක් පමණ වාර්තා වන්නේ බස්නාහිර පළාතෙන් බව වසංගත රෝග විද්‍යා ඒකකය පවසයි
මම විස්තර අසමි වේටර් අයියගෙන් කියා
2009 පටන් ටිබට් දේශයේ දියත් වෙමින් ඇති චීන විරෝධී රැල්ලේදී හැට දෙනෙක් පමණ ගිනි තබා ගෙන තිබේ ඔවුන්ගෙන් වැඩි දෙනා ජීවිතක්ෂයට පත් වූ බවයි ප්‍රකාශ වන්නේ
හේමාල් රණසිංහ ගම්‍යා විජේදාස මහේන්ද්‍ර පෙරේරා බිමල් ජයකොඩි අශාන් ඩයස් සඳලි ප්‍රනාන්දු සරත් කොතලාවල සමන් හේමරත්න අසංග පෙරේරා නන්දන හෙට්ටිආරච්චි ඇතුළු පිරිසක් චිත්‍රපටය සඳහා රංගන දායකත්වය ලබා දෙයි
ඇය රජුගේ රාජ්‍යේශ පදවීන්හි ස්ත්‍රීරූපී නාම දැරුව ද ඓතිහාසිකව රජුගේ දේශපාලන සහ යුධමය බලයට හිමිකම් කියන්නේ නැත
මට නං බං ඉදිරිපසින් කැටයම කපන්න අමාරුයි
ජනපති සිරිසේන මහතා තම කැබිනට් අමාත්‍ය කණ්ඩායම හමුවේ කියා සිටි අයුරින්ම මත්කුඩු යනු විශාල වශයෙන් මුදල් ගැවසෙන ක්ෂේත‍්‍රයකි
ඊට වඩා බාහිර ක්‍රියාකාරකම් වලට තමයි මම යොමු වුණේ
ජනවාරි 23 මැක්ස් ටිවී මාධ්‍යවේදීන්ට පහරදීම
අන්තිමට තෝරගන්නෙ එක්කෙනයි
ප්‍රබන්ධ යථාර්ථයක් නොවන කායිකතාව ස්පර්ශ කළ හැකි ප්‍රතිරූප සහ දේශපාලන ගුප්තාර්ථය අතින් ස්වරූප නරඹන කෙනකුට තර්කොව්ස්කි සිහිපත් වෙයි
රටේ සියලුම ජනතාවට ඊ කාඩ් පතක් ලෝකයේ නවීනතම වෛද්‍ය ක්‍රමයෙන් ප්‍රතිකාර ඇරඹීමේ සූදානමක්
අර රාජ් දී ඇති ලින්ක් එකේ අදහස් හුමාරුව කියවන්න
සල්ලි පස්සේම දුවනවා මිසක් හොඳ සාරගර්භ නිර්මාණයක් එළිදැක්වීමේ අරමුණක් ඔවුන් තුළ පෙනෙන්නට නැහැ
තුළ වෙබ් යෙදුම් අතර සන්නිවේදනය කළමනාකරණය කිරීම සඳහා පේලි සහ සේවා බස් භාවිතා කිරීම
මෙවැනි ක්‍රියාවලට වගකිව යුත්තේ පැහැදිලිව ම මගීන් ය
මට සහ මගෙන් පසු ඉතිරිවන පරම්පරාවට මේ නිවස නඩත්තු කිරීමට හැකි වෙන ආකාරයේ සරල එකක්ද
එහි හෝ ගොනුවේ ඇති යන අයිකනය ධාවනය කිරීමේන් යාවත්කාලීන කරගත හැකි
තමා ශ්‍රේෂ්ඨාධිකරණයෙන් මත විමසුමක් සිදු කළේ තමාට ජනාධිපති ධූරයේ රැදී සිටීමට තිබෙන වසර ගණන පිළිබඳ නොව 19වන ව්‍යවස්ථා සංශෝධනය සමග වර්තමාන ජනාධිපතිවරයාගේ නිල කාලය පිළිබඳසමාජය තුළ මත දෙකක් ඉදිරිපත්වී තිබීම නිසා බව ද ජනාධිපතිතුමා මෙහිදී කියා සිටියේය
මගේ මතු බුදුවෙන මෑණියන්ගෙ දහසක් ගුණකඳ ගායනා කරමින් තුන්දෙනා එළිවෙනකන් ටොයිලට් එක බදුඅරන් වගේ පදිංචිවෙලා හිටියා
යකෝ මේ යකා ජෙප්පෙක් නේද මෙන්න මූ මට වහ පෙව්වෝ කියාගෙන උපවාසෙන් නැගිට්ට විමල් එකපාරටම දැක්කේ මුණ දිහා කන්න වගේ බලාගෙන ඉන්න සශිව.රට පුරා සවරමේ ගිහින් බේගල් ඇදබාල එනවා මෙතන
ශ්‍රී ලංකා පිල ලකුණු 34 කින් ජයගනී
බ්ලා බ්ලා බ්ලා බ්ලා වෙනකොට හයිවේ විවෘත කරන හැටි වරායවල් විවෘත වන හැටි බලාගෙන ඉන්න හැටි මට කල්පනා කරන්න වෙනව.
උන්වහන්සේ ක‍්‍රිස්තු වර්ෂ 410 දී ලංකාවට වැඩම කරලා අවුරුදු දෙකක් අනුරාධපුරයේ වාසය කළා
අද ආරම්භවන තරගය අතරතුර අපේක්ෂා නොකරන අයුරින් වරින් වර ඇද හැලෙන වර්ෂාව හේතුවෙන් වසා දැමෙන තණතීරුව හා අවට වායුගෝලයේ ආර්ද්‍රතාව දකුණු අප්‍රිකානු වේග පන්දු යවන්නන්ට වැඩි වාසි උකහාගැනීමට අවස්ථාව උදාකර දෙනු ඇතැයි විශ්වාස කෙරේ
සුවාතත් මාර මීටරයක් තමයි
ධර්මතාවය ඉස්සරහට එනවා
ඒත් තනි තනි ඕනකං වෙනස්
මහේන්ද්‍ර සිං දෝනි නොදැවී ලබාගත් ලකුණු ගණන 32කි

"""

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""
    def __init__(self, vocab_file, do_lower_case=True):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def load_vocab():
    # makes segmenter instance and loads the model file (m.model)
    sp = spm.SentencePieceProcessor()
    sp.load('m-uni.model')

    # encode: text => id
    print(sp.encode_as_pieces('එවැනි ගිවිසුමක් ලංකා ආණ්ඩුව'))
    print(sp.encode_as_ids('එවැනි ගිවිසුමක් ලංකා ආණ්ඩුව'))

    # decode: id => text
    print(sp.decode_pieces(['▁This', '▁is', '▁a', '▁t', 'est']))
    print(sp.decode_ids([209, 31, 9, 375, 586]))

if __name__ == '__main__':
    inFile = open("/home/rumesh/Downloads/FYP/datasets/15m-tokenized/tokenized_shard_100000.txt")
    outFile = open("out.txt","w")
    inData = inFile.readlines()[:50]
    outList = []
    basic_token = BasicTokenizer(do_lower_case=False)
    vocab = set()
    wordpiece_token = WordpieceTokenizer(vocab=vocab,unk_token="UNK")
    for line in inData:
        tokens = basic_token.tokenize(line)
        vocab.update(set(tokens))
        line = ' '.join(tokens)
        wp_tokens = wordpiece_token.tokenize(line)
        line = ' '.join(wp_tokens)
        outFile.write(line+"\n")
    # print(len(vocab))
    load_vocab()

