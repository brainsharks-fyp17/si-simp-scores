from typing import List, Text, Tuple

# import model.Constants as Constants
# import logging
# import codecs
# import pickle
import nltk
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch.nn as nn

# logger = logging.getLogger()
# embedding shape: torch.Size([30522, 512])
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from tqdm import tqdm


class Params:
    vocab_size = 30000
    dropout = 0.1
    PAD = 0


class LanguageModel(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, dropout_p, pad_idx):
        super(LanguageModel, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.pad_idx = pad_idx

        # num_embeddings = vocabulary size
        self.embedding = nn.Embedding(num_embeddings=output_size, embedding_dim=hidden_size, padding_idx=self.pad_idx)

        self.rnn = nn.LSTM(self.emb_size, self.hidden_size, num_layers=2, dropout=dropout_p, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

        self.fc = nn.Linear(hidden_size, self.output_size)

    def forward(self, input_seq, tgt_seq):
        input_seq = self.dropout(self.embedding(input_seq))
        out, hidden = self.rnn(input_seq, None)

        out = self.fc(self.dropout(out))
        loss_func = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        out = out.view(-1, out.size(-1))
        tgt_seq = tgt_seq.view(-1)

        loss = loss_func(out, tgt_seq)

        return loss


def train():
    pass


class Vocabulary:
    def __init__(self, file_name: Text, mask_index=0):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(file_name)
        self.mask_index = mask_index

    def get_ids(self, sentence: Text) -> List[int]:
        return self.sp.encode_as_ids(sentence)

    def get_sentence(self, ids: List[int]) -> Text:
        return self.sp.decode_ids(ids)

    def _get_pieces(self, sentence: Text) -> List[Text]:
        return self.sp.encode_as_pieces(sentence)

    def _get_ids_for_pieces(self, ids: List[Text]) -> List[int]:
        return self.sp.decode_pieces(ids)


class Vectorizer:
    def __init__(self, vocab: Vocabulary, length=100):
        self.vocab = vocab
        self.sentence_length = length

    def vectorize(self, sentence: Text) -> Tuple[np.array, np.array]:
        ids = self.vocab.get_ids("<s>" + sentence + "</s>")
        if len(ids) > 100:
            return None, None
        ids1, ids2 = ids[:], ids[:]  # copy the lists
        del ids1[len(ids) - 2]  # remove from the back
        del ids2[2]  # remove from the front

        x = np.zeros(self.sentence_length, dtype=np.int64)
        x[:len(ids1)] = ids1

        y = np.zeros(self.sentence_length, dtype=np.int64)
        y[:len(ids2)] = ids2

        if self.vocab.mask_index != 0:
            x[len(ids1):] = self.vocab.mask_index
            y[len(ids2):] = self.vocab.mask_index

        return x, y


class CBOWDataset(Dataset):
    def __init__(self, vectorizer: Vectorizer, data_file, train_prop, val_prop):
        self.vectorizer = vectorizer
        self.num_lines = sum(1 for _ in open(data_file))
        self.train_ids = []
        self.test_ids = []
        self.validation_ids = []

        for row_num in range(self.num_lines):
            if row_num <= self.num_lines * train_prop:
                self.train_ids.append(row_num)
            elif (row_num > self.num_lines * train_prop) and \
                    (row_num <= self.num_lines * train_prop + self.num_lines * val_prop):
                self.validation_ids.append(row_num)
            else:
                self.test_ids.append(row_num)

    @staticmethod
    def create_cbow_csv(data_file: Text, out_file_name="cbow_out.tsv", train_proportion=0.7, val_proportion=0.15,
                        window_size=5) -> None:
        MASK_TOKEN = "<MASK>"
        cleaned_sentences = [line.strip() for line in open(data_file).readlines()]

        def flatten(outer_list):
            return [item for inner_list in outer_list for item in inner_list]

        windows = flatten([list(nltk.ngrams([MASK_TOKEN] * window_size + sentence.split(' ') + \
                                            [MASK_TOKEN] * window_size, window_size * 2 + 1)) \
                           for sentence in tqdm(cleaned_sentences)])

        # Create cbow data
        data = []
        for window in tqdm(windows):
            target_token = window[window_size]
            context = []
            for i, token in enumerate(window):
                if token == MASK_TOKEN or i == window_size:
                    continue
                else:
                    context.append(token)
            data.append([' '.join(token for token in context), target_token])
        # Convert to dataframe
        cbow_data = pd.DataFrame(data, columns=["context", "target"])
        # Create split data
        n = len(cbow_data)

        def get_split(row_num):
            if row_num <= n * train_proportion:
                return 'tr'
            elif (row_num > n * train_proportion) and (row_num <= n * train_proportion + n * val_proportion):
                return 'va'
            else:
                return 'te'

        cbow_data['split'] = cbow_data.apply(lambda row: get_split(row.name), axis=1)
        cbow_data.to_csv(out_file_name, index=False, sep='\t', index_label=False, header=False)

    def __getitem__(self, index) -> T_co:
        pass


if __name__ == '__main__':
    # vo = Vocabulary("/home/rumesh/Downloads/FYP/sentencepiece-vocabs/m-30000-bpe.model")
    # sen = "<s>ඩෙංගු රෝගීන්ගෙන් ආසන්න වශයෙන් 423 ක් පමණ වාර්තා වන්නේ බස්නාහිර</s>"
    # print(vo._get_pieces(sen))
    # print(vo.get_ids(sen))
    # vec = Vectorizer(vocab=vo, length=100)
    # x, y = vec.vectorize(sen)
    # print(len(x), x)
    # print(len(y), y)
    # # print(vo.get_ids(sen))
    # print(vo.get_sentence(x.tolist()))
    # print(vo.get_sentence(y.tolist()))
    # ids = [29925, 5, 4986, 4855, 491, 1948, 749, 7186, 29983, 121, 407, 770, 705, 4113, 0,0,0,0,0]
    # print(vo.get_sentence(ids))

    csv = CBOWDataset.create_cbow_csv('in.txt')
    print(csv)
