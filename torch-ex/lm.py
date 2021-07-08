import linecache
import os
from typing import List, Text

import nltk
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
from tqdm import tqdm


class args:
    # Data and Path information
    model_state_file = "model.pth"
    vocab_file = "/home/rumesh/Downloads/FYP/sentencepiece-vocabs/m-30000-bpe.model"
    data_file = "/home/rumesh/Downloads/FYP/datasets/15m-tokenized/tokenized_shard_100000.txt"
    train_data_file = "cbow_out_train.tsv"
    validation_data_file = "cbow_out_validation.tsv"
    test_data_file = "cbow_out_test.tsv"
    save_dir = "model_storage/cbow-si"
    window_size = 25
    sentence_length = 25
    # Model hyper parameters
    num_embeddings = 30000
    embedding_dim = 300
    rnn_hidden_size = 256
    output_size = 10
    lstm_layers = 2
    dropout_p = 0.5
    pad_idx = 0
    # Training hyper parameters
    seed = 1337
    num_epochs = 25
    learning_rate = 0.002
    batch_size = 50
    early_stopping_criteria = 5
    # Runtime options
    cuda = False
    catch_keyboard_interrupt = True
    reload_from_files = False
    expand_filepaths_to_save_dir = True


class LanguageModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, rnn_hidden_size, lstm_layers, output_size, dropout_p, pad_idx):
        super(LanguageModel, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.lstm_layers = lstm_layers
        self.pad_idx = pad_idx

        # num_embeddings = vocabulary size
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                                      padding_idx=self.pad_idx)

        self.rnn = nn.LSTM(self.embedding_dim, self.rnn_hidden_size, num_layers=lstm_layers, dropout=dropout_p,
                           batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

        self.fc = nn.Linear(rnn_hidden_size, num_embeddings)

    def forward(self, x_in, apply_softmax=False):
        embed_out = self.embedding(x_in)
        embed_out = self.dropout(embed_out)
        out, hidden = self.rnn(embed_out, None)

        y_out = self.fc(self.dropout(out))

        # y_out=y_out.view(-1)
        y_out = y_out.sum(dim=1)
        # print(y_out.shape)

        # loss_func = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        # out = out.view(-1, out.size(-1))
        # tgt_seq = tgt_seq.view(-1)
        #
        # loss = loss_func(out, tgt_seq)
        # apply_softmax = True
        if apply_softmax:
            y_out = F.softmax(y_out, dim=1)

        return y_out


class Vocabulary:
    def __init__(self, file_name: Text, pad_index=0):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(file_name)
        self.pad_index = pad_index

    def get_ids(self, sentence: Text) -> List[int]:
        return self.sp.encode_as_ids(sentence)

    def get_sentence(self, ids: List[int]) -> Text:
        return self.sp.decode_ids(ids)

    def _get_pieces(self, sentence: Text) -> List[Text]:
        return self.sp.encode_as_pieces(sentence)

    def _get_ids_for_pieces(self, ids: List[Text]) -> List[int]:
        return self.sp.decode_pieces(ids)


class Vectorizer:
    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab

    def vectorize_to_length(self, sentence: Text, sentence_length) -> np.array:
        ids = self.vocab.get_ids(sentence)
        if len(ids) > sentence_length:
            return None

        x = np.zeros(sentence_length, dtype=np.int64)
        x[:len(ids)] = ids

        if self.vocab.pad_index != 0:
            x[len(ids):] = self.vocab.pad_index

        return x

    def vectorize(self, sentence: Text) -> List[str]:
        return [str(i) for i in self.vocab.get_ids(sentence)]

    def fill_with_mask(self, lst: List, sentence_length: int) -> np.array:
        x = np.zeros(sentence_length, dtype=np.int64)
        x[:len(lst)] = lst

        if self.vocab.pad_index != 0:
            x[len(lst):] = self.vocab.pad_index

        return x


class CBOWDataset(Dataset):
    def __getitem__(self, index) -> np.array:
        try:
            line = linecache.getline(filename=self.data_file, lineno=index).strip().split("\t")
            target = int(line[1])
            lst = line[0].split(" ")
            lst = [int(i) for i in lst]
            return {'x_data': self.vectorizer.fill_with_mask(lst, self.sentence_length),
                    'y_target': target}
        except IndexError:
            print("Index error at", index)
            return {'x_data': self.vectorizer.fill_with_mask([0, 0, 0, 0], self.sentence_length),
                    'y_target': 0}

    def __init__(self, vectorizer: Vectorizer, data_file, sentence_length):
        self.vectorizer = vectorizer
        self.train_ids = []
        self.test_ids = []
        self.validation_ids = []
        self.current_idx = 0
        self.data_file = data_file
        self.sentence_length = sentence_length
        with open(data_file) as file:
            for i, line in enumerate(file):
                line = line.strip().split("\t")
                if len(line) != 3:
                    print("Error in line")
                    continue
                if line[2] == 'tr':
                    self.train_ids.append(i)
                elif line[2] == 'va':
                    self.validation_ids.append(i)
                else:
                    self.test_ids.append(i)
        self.file = open(data_file)
        print(len(self))

    @staticmethod
    def create_cbow_csv(vectorizer: Vectorizer,
                        data_file: Text,
                        train_proportion=0.7,
                        val_proportion=0.15,
                        window_size=5) -> None:
        """
        :param vectorizer: Vectorizer
        :param data_file: path of the file that contains sentences. One line should contain only one sentence.
        Data is assumed been cleaned.
        :param train_proportion: train proportion
        :param val_proportion: validation proportion
        :param window_size: window size
        :return: None
        """
        MASK_TOKEN = "-50"
        cleaned_sentences = [line.strip() for line in open(data_file).readlines()]

        def flatten(outer_list):
            return [item for inner_list in outer_list for item in inner_list]

        windows = flatten([list(nltk.ngrams([MASK_TOKEN] * window_size + vectorizer.vectorize(sentence) +
                                            [MASK_TOKEN] * window_size, window_size * 2 + 1))
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
        cbow_data_train = cbow_data.loc[cbow_data['split'] == 'tr']
        cbow_data_train.to_csv(args.train_data_file, index=False, sep='\t', index_label=False, header=False,
                               mode="w")

        cbow_data_test = cbow_data.loc[cbow_data['split'] == 'te']
        cbow_data_test.to_csv(args.test_data_file, index=False, sep='\t', index_label=False, header=False,
                              mode="w")

        cbow_data_validation = cbow_data.loc[cbow_data['split'] == 'va']
        cbow_data_validation.to_csv(args.validation_data_file, index=False, sep='\t', index_label=False,
                                    header=False,
                                    mode="w")

    def __len__(self):
        return len(self.test_ids) + len(self.train_ids) + len(self.validation_ids)

    def num_batches(self) -> int:
        return len(self) // args.batch_size

    @staticmethod
    def set_split(param):
        print("Loaded dataset", param)


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}


def update_train_state(args, model, train_state):
    """Handle the training state updates.

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]

        # If loss worsened
        if loss_t >= train_state['early_stopping_best_val']:
            # Update step
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state


def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def generate_batches(dataset: Dataset, batch_size: int, shuffle=True,
                     drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


if args.expand_filepaths_to_save_dir:
    # args.vectorizer_file = os.path.join(args.save_dir,
    #                                     args.vectorizer_file)

    args.model_state_file = os.path.join(args.save_dir,
                                         args.model_state_file)

    print("Expanded filepaths: ")
    # print("\t{}".format(args.vectorizer_file))
    print("\t{}".format(args.model_state_file))
# Check CUDA
if not torch.cuda.is_available():
    args.cuda = False
args.device = torch.device("cuda" if args.cuda else "cpu")
print("Using CUDA: {}".format(args.cuda))
# Set seed for reproducibility
set_seed_everywhere(args.seed, args.cuda)
# handle dirs
handle_dirs(args.save_dir)

vocab = Vocabulary(args.vocab_file)
vec = Vectorizer(vocab=vocab)
dataset_train = CBOWDataset(vectorizer=vec, data_file=args.train_data_file,
                            sentence_length=args.sentence_length)
# dataset_test = CBOWDataset(vectorizer=vec, data_file='cbow_out_test.tsv', batch_size=32, sentence_length=12)
dataset_validation = CBOWDataset(vectorizer=vec, data_file=args.validation_data_file,
                                 sentence_length=args.sentence_length)
model = LanguageModel(num_embeddings=args.num_embeddings, embedding_dim=args.embedding_dim,
                      rnn_hidden_size=args.rnn_hidden_size, output_size=args.output_size,
                      dropout_p=args.dropout_p, pad_idx=args.pad_idx, lstm_layers=args.lstm_layers)
print(model)


def train(model):
    lang_model = model.to(args.device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lang_model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     mode='min', factor=0.5,
                                                     patience=1)
    train_state = make_train_state(args)
    epoch_bar = tqdm(desc='training routine',
                     total=args.num_epochs,
                     position=0)
    try:
        for epoch_index in range(args.num_epochs):
            train_state['epoch_index'] = epoch_index

            # Iterate over training dataset

            # setup: batch generator, set loss and acc to 0, set train mode on

            dataset_train.set_split('train')
            train_bar = tqdm(desc='split=train',
                             total=dataset_train.num_batches(),
                             position=1,
                             leave=True)
            dataset_validation.set_split('val')
            val_bar = tqdm(desc='split=val',
                           total=dataset_validation.num_batches(),
                           position=1,
                           leave=True)
            batch_generator_train = generate_batches(dataset_train,
                                                     batch_size=args.batch_size,
                                                     device="cuda" if args.cuda else "cpu")
            running_loss = 0.0
            running_acc = 0.0
            lang_model.train()

            for batch_index, batch_dict in enumerate(batch_generator_train):
                # the training routine is these 5 steps:

                # --------------------------------------
                # step 1. zero the gradients
                optimizer.zero_grad()

                # step 2. compute the output
                y_pred = lang_model(x_in=batch_dict['x_data'])

                # step 3. compute the loss
                loss = loss_func(y_pred, batch_dict['y_target'])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # step 4. use loss to produce gradients
                loss.backward()

                # step 5. use optimizer to take gradient step
                optimizer.step()
                # clear the linecache
                # linecache.clearcache()
                # -----------------------------------------
                # compute the accuracy
                acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                # update bar
                train_bar.set_postfix(loss=running_loss, acc=running_acc,
                                      epoch=epoch_index)
                train_bar.update()

            train_state['train_loss'].append(running_loss)
            train_state['train_acc'].append(running_acc)

            # Iterate over val dataset

            # setup: batch generator, set loss and acc to 0; set eval mode on
            dataset_validation.set_split('val')
            batch_generator_validation = generate_batches(dataset_validation,
                                                          batch_size=args.batch_size,
                                                          device="cuda" if args.cuda else "cpu")
            running_loss = 0.
            running_acc = 0.
            lang_model.eval()

            for batch_index, batch_dict in enumerate(batch_generator_validation):
                # compute the output
                y_pred = lang_model(x_in=batch_dict['x_data'])

                # step 3. compute the loss
                loss = loss_func(y_pred, batch_dict['y_target'])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # compute the accuracy
                acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)
                val_bar.set_postfix(loss=running_loss, acc=running_acc,
                                    epoch=epoch_index)
                val_bar.update()

            linecache.clearcache()
            train_state['val_loss'].append(running_loss)
            train_state['val_acc'].append(running_acc)

            train_state = update_train_state(args=args, model=lang_model,
                                             train_state=train_state)

            scheduler.step(train_state['val_loss'][-1])

            if train_state['stop_early']:
                break

            train_bar.n = 0
            val_bar.n = 0
            epoch_bar.update()
    except KeyboardInterrupt:
        print("Exiting loop")


def create_split_datasets():
    vo = Vocabulary(args.vocab_file)
    vec = Vectorizer(vocab=vo)
    CBOWDataset.create_cbow_csv(vectorizer=vec,
                                data_file=args.data_file,
                                window_size=args.window_size
                                )


if __name__ == '__main__':
    # vo = Vocabulary(args.vocab_file)
    # vec = Vectorizer(vocab=vo)
    # data_set = CBOWDataset(vectorizer=vec, data_file=args.test_data_file, sentence_length=args.sentence_length)
    # batch_gen = generate_batches(dataset=data_set, batch_size=32)
    # b1 = next(batch_gen)
    # var = b1['x_data'].shape
    # print(var)
    # train(model)
    create_split_datasets()
