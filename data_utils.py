import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from transformers import BertTokenizer
from torch.utils.data import Dataset


def ParseData(data_path):
    with open(data_path) as infile:
        all_data = []
        data = json.load(infile)
        for d in data:
            for aspect in d['aspects']:
                text_list = list(d['token'])
                tok = list(d['token'])  # word token
                length = len(tok)  # real length
                # if args.lower == True:
                tok = [t.lower() for t in tok]
                tok = ' '.join(tok)
                asp = list(aspect['term'])  # aspect
                asp = [a.lower() for a in asp]
                asp = ' '.join(asp)
                label = aspect['polarity']  # label
                pos = list(d['pos'])  # pos_tag
                head = list(d['head'])  # head
                deprel = list(d['deprel'])  # deprel
                # position
                aspect_post = [aspect['from'], aspect['to']]
                post = [i - aspect['from'] for i in range(aspect['from'])] \
                       + [0 for _ in range(aspect['from'], aspect['to'])] \
                       + [i - aspect['to'] + 1 for i in range(aspect['to'], length)]
                # aspect mask
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]
                else:
                    mask = [0 for _ in range(aspect['from'])] \
                           + [1 for _ in range(aspect['from'], aspect['to'])] \
                           + [0 for _ in range(aspect['to'], length)]

                sample = {'text': tok, 'aspect': asp, 'pos': pos, 'post': post, 'head': head,
                          'deprel': deprel, 'length': length, 'label': label, 'mask': mask,
                          'aspect_post': aspect_post, 'text_list': text_list}
                all_data.append(sample)

    return all_data


def reshape_tree(obj, max_hop=5, bert=False):
    if bert:
        text = obj['text_list']
    else:
        text = obj['text']
        text = text.split()
    aspect_start, aspect_end = obj['aspect_post']
    dependency = obj['deprel']
    head = obj['head']
    dependencies = list(zip(dependency, head, list(range(1, len(head) + 1)), text))

    dep_tag = []
    dep_idx = []
    dep_dir = []

    for i in range(aspect_start, aspect_end):
        for dep in dependencies:
            if i == dep[1] - 1:
                if (dep[2] - 1 < aspect_start or dep[2] - 1 >= aspect_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':
                        dep_tag.append(dep[0])
                        dep_dir.append(1)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[2] - 1)
            elif dependencies[i][1] == dep[2]:
                if (dep[2] - 1 < aspect_start or dep[2] - 1 >= aspect_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_dir.append(2)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[2] - 1)


    current_hop = 2
    added = True
    while current_hop <= max_hop and len(dep_idx) < len(text) and added:
        added = False
        dep_idx_temp = deepcopy(dep_idx)
        for i in dep_idx_temp:
            for dep in dependencies:
                if i == dep[1] - 1:
                    if (dep[2] - 1 < aspect_start or dep[2] - 1 >= aspect_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                        if str(dep[0]) != 'punct':
                            dep_tag.append('ncon_' + str(current_hop))
                            dep_dir.append(1)
                        else:
                            dep_tag.append('<pad>')
                            dep_dir.append(0)
                        dep_idx.append(dep[2] - 1)
                elif dependencies[i][1] == dep[2]:
                    if (dep[2] - 1 < aspect_start or dep[2] - 1 >= aspect_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                        if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                            dep_tag.append('ncon_' + str(current_hop))
                            dep_dir.append(2)
                        else:
                            dep_tag.append('<pad>')
                            dep_dir.append(0)
                        dep_idx.append(dep[2] - 1)
                        added = True
        current_hop += 1

    for idx, token in enumerate(text):
        if idx not in dep_idx and (idx < aspect_start or idx >= aspect_end):
            dep_tag.append('non-connect')
            dep_dir.append(0)
            dep_idx.append(idx)

    # aspect padding
    for idx, token in enumerate(text):
        if idx not in dep_idx:
            dep_tag.append('<pad>')
            dep_dir.append(0)
            dep_idx.append(idx)

    index = [i[0] for i in sorted(enumerate(dep_idx), key=lambda x: x[1])]
    dep_tag = [dep_tag[i] for i in index]
    dep_idx = [dep_idx[i] for i in index]
    dep_dir = [dep_dir[i] for i in index]

    assert len(text) == len(dep_idx), 'length wrong'

    return dep_tag


def build_tokenizer(fnames, max_length, data_file):
    parse = ParseData
    if os.path.exists(data_file):
        print('loading tokenizer:', data_file)
        tokenizer = pickle.load(open(data_file, 'rb'))
    else:
        tokenizer = Tokenizer.from_files(fnames=fnames, max_length=max_length, parse=parse)
        pickle.dump(tokenizer, open(data_file, 'wb'))
    return tokenizer


class Vocab(object):
    ''' vocabulary of dataset '''

    def __init__(self, vocab_list, add_pad, add_unk):
        self._vocab_dict = dict()
        self._reverse_vocab_dict = dict()
        self._length = 0
        if add_pad:
            self.pad_word = '<pad>'
            self.pad_id = self._length
            self._length += 1
            self._vocab_dict[self.pad_word] = self.pad_id
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = self._length
            self._length += 1
            self._vocab_dict[self.unk_word] = self.unk_id
        for w in vocab_list:
            self._vocab_dict[w] = self._length
            self._length += 1
        for w, i in self._vocab_dict.items():
            self._reverse_vocab_dict[i] = w

    def word_to_id(self, word):
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]

    def id_to_word(self, id_):
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(id_, self.unk_word)
        return self._reverse_vocab_dict[id_]

    def has_word(self, word):
        return word in self._vocab_dict

    def __len__(self):
        return self._length

    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


class Tokenizer(object):
    ''' transform text to indices '''

    def __init__(self, vocab, max_length, lower, pos_char_to_int, pos_int_to_char):
        self.vocab = vocab
        self.max_length = max_length
        self.lower = lower

        self.pos_char_to_int = pos_char_to_int
        self.pos_int_to_char = pos_int_to_char

    @classmethod
    def from_files(cls, fnames, max_length, parse, lower=True):
        corpus = set()
        pos_char_to_int, pos_int_to_char = {}, {}
        for fname in fnames:
            for obj in parse(fname):
                text_raw = obj['text']
                if lower:
                    text_raw = text_raw.lower()
                corpus.update(Tokenizer.split_text(text_raw))
        return cls(vocab=Vocab(corpus, add_pad=True, add_unk=True), max_length=max_length, lower=lower,
                   pos_char_to_int=pos_char_to_int, pos_int_to_char=pos_int_to_char)

    @staticmethod
    def pad_sequence(sequence, pad_id, maxlen, dtype='int64', padding='post', truncating='post'):
        x = (np.zeros(maxlen) + pad_id).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = Tokenizer.split_text(text)
        sequence = [self.vocab.word_to_id(w) for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence.reverse()
        return Tokenizer.pad_sequence(sequence, pad_id=self.vocab.pad_id, maxlen=self.max_length,
                                      padding=padding, truncating=truncating)

    @staticmethod
    def split_text(text):
        # for ch in ["\'s", "\'ve", "n\'t", "\'re", "\'m", "\'d", "\'ll", ",", ".", "!", "*", "/", "?", "(", ")", "\"", "-", ":"]:
        #     text = text.replace(ch, " "+ch+" ")
        return text.strip().split()


class SentenceDataset(Dataset):
    ''' PyTorch standard dataset class '''

    def __init__(self, fname, tokenizer, opt, vocab_help):

        parse = ParseData
        post_vocab, pos_vocab, dep_vocab, dep_tag_vocab, pol_vocab = vocab_help
        data = list()
        polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
        special_list = ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "RB",
                        "RBR", "RBS"]

        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
            text = tokenizer.text_to_sequence(obj['text'])
            aspect = tokenizer.text_to_sequence(obj['aspect'])  # max_length=10
            post = [post_vocab.stoi.get(t, post_vocab.unk_index) for t in obj['post']]
            post = tokenizer.pad_sequence(post, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post',
                                          truncating='post')
            pos = [pos_vocab.stoi.get(t, pos_vocab.unk_index) for t in obj['pos']]
            pos = tokenizer.pad_sequence(pos, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post',
                                         truncating='post')
            pos_mask = [1 if t in special_list else 0 for t in obj['pos']]
            pos_mask = tokenizer.pad_sequence(pos_mask, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64',
                                              padding='post',
                                              truncating='post')
            deprel = [dep_vocab.stoi.get(t, dep_vocab.unk_index) for t in obj['deprel']]
            deprel = tokenizer.pad_sequence(deprel, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64',
                                            padding='post', truncating='post')
            mask = tokenizer.pad_sequence(obj['mask'], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64',
                                          padding='post', truncating='post')

            adj = np.ones(opt.max_length) * opt.pad_id

            if opt.reshape:
                dep_tag = reshape_tree(obj)
                dep_tag = [dep_tag_vocab.stoi.get(t, dep_tag_vocab.unk_index) for t in dep_tag]
                dep_tag = tokenizer.pad_sequence(dep_tag, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64',
                                                 padding='post', truncating='post')

            length = obj['length']
            polarity = polarity_dict[obj['label']]
            data.append({
                'text_aspect': [obj['text'], obj['aspect']],
                'text': text,
                'aspect': aspect,
                'post': post,
                'pos': pos,
                'deprel': deprel,
                'deptag': dep_tag,
                'pos_mask': pos_mask,
                'mask': mask,
                'length': length,
                'polarity': polarity
            })

        self._data = data

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)


def _load_wordvec(data_path, embed_dim, vocab=None):
    with open(data_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        word_vec = dict()
        if embed_dim == 200:
            for line in f:
                tokens = line.rstrip().split()
                if tokens[0] == '<pad>' or tokens[0] == '<unk>':  # avoid them
                    continue
                if vocab is None or vocab.has_word(tokens[0]):
                    word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        elif embed_dim == 300:
            for line in f:
                tokens = line.rstrip().split()
                if tokens[0] == '<pad>':  # avoid them
                    continue
                elif tokens[0] == '<unk>':
                    word_vec['<unk>'] = np.random.uniform(-0.25, 0.25, 300)
                word = ''.join((tokens[:-300]))
                if vocab is None or vocab.has_word(tokens[0]):
                    word_vec[word] = np.asarray(tokens[-300:], dtype='float32')
        else:
            print("embed_dim error!!!")
            exit()

        return word_vec


def build_embedding_matrix(vocab, embed_dim, data_file):
    if os.path.exists(data_file):
        print('loading embedding matrix:', data_file)
        embedding_matrix = pickle.load(open(data_file, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(vocab), embed_dim))
        fname = './DualGCN/glove/glove.840B.300d.txt'
        word_vec = _load_wordvec(fname, embed_dim, vocab)
        for i in range(len(vocab)):
            vec = word_vec.get(vocab.id_to_word(i))
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(data_file, 'wb'))
    return embedding_matrix


def softmax(x):
    if len(x.shape) > 1:
        # matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x


class Tokenizer4BertGCN:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

    def tokenize(self, s):
        return self.tokenizer.tokenize(s)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)


class ABSAGCNData(Dataset):
    def __init__(self, fname, tokenizer, opt, dep_tag_vocab):
        self.data = []
        parse = ParseData
        polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
        special_list = ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "RB",
                        "RBR", "RBS"]
        self.max_len = 0
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
            new_dep_tag = None
            polarity = polarity_dict[obj['label']]
            term_start = obj['aspect_post'][0]
            term_end = obj['aspect_post'][1]
            text_list = obj['text_list']
            pos_mask = [1 if t in special_list else 0 for t in obj['pos']]
            self.max_len = max(self.max_len, len(text_list))
            left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end:]

            if opt.reshape:
                dep_tag = reshape_tree(obj, bert=True)
                assert len(dep_tag) == len(text_list)

            left_tokens, term_tokens, right_tokens = [], [], []
            left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], []

            for ori_i, w in enumerate(left):
                for t in tokenizer.tokenize(w):
                    left_tokens.append(t)  # * ['expand', '##able', 'highly', 'like', '##ing']
                    left_tok2ori_map.append(ori_i)  # * [0, 0, 1, 2, 2]
            asp_start = len(left_tokens)
            offset = len(left)
            for ori_i, w in enumerate(term):
                for t in tokenizer.tokenize(w):
                    term_tokens.append(t)
                    term_tok2ori_map.append(ori_i + offset)
            asp_end = asp_start + len(term_tokens)
            offset += len(term)
            for ori_i, w in enumerate(right):
                for t in tokenizer.tokenize(w):
                    right_tokens.append(t)
                    right_tok2ori_map.append(ori_i + offset)

            while len(left_tokens) + len(right_tokens) > tokenizer.max_seq_len - 2 * len(term_tokens) - 3:
                if len(left_tokens) > len(right_tokens):
                    left_tokens.pop(0)
                    left_tok2ori_map.pop(0)
                else:
                    right_tokens.pop()
                    right_tok2ori_map.pop()

            bert_tokens = left_tokens + term_tokens + right_tokens
            tok2ori_map = left_tok2ori_map + term_tok2ori_map + right_tok2ori_map

            if opt.reshape:
                new_dep_tag = [dep_tag[i] for i in tok2ori_map]
                new_dep_tag = [dep_tag_vocab.stoi.get(t, dep_tag_vocab.unk_index) for t in new_dep_tag]
                assert len(new_dep_tag) == len(bert_tokens)
            pos_mask = [pos_mask[i] for i in tok2ori_map]

            context_asp_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(
                bert_tokens) + [tokenizer.sep_token_id] + tokenizer.convert_tokens_to_ids(term_tokens) + [
                                  tokenizer.sep_token_id]
            context_asp_len = len(context_asp_ids)
            paddings = [0] * (tokenizer.max_seq_len - context_asp_len)
            context_len = len(bert_tokens)
            context_asp_seg_ids = [0] * (1 + context_len + 1) + [1] * (len(term_tokens) + 1) + paddings
            src_mask = [0] + [1] * context_len + [0] * (opt.max_length - context_len - 1)
            aspect_mask = [0] + [0] * asp_start + [1] * (asp_end - asp_start)
            aspect_mask = aspect_mask + (opt.max_length - len(aspect_mask)) * [0]
            context_asp_attention_mask = [1] * context_asp_len + paddings
            context_asp_ids += paddings
            if opt.reshape:
                new_dep_tag = [dep_tag_vocab.stoi.get('<pad>')] + new_dep_tag + [dep_tag_vocab.stoi.get('<pad>')] \
                              * (1 + len(term_tokens) + 1)
                new_dep_tag += [dep_tag_vocab.stoi.get('<pad>')] * (tokenizer.max_seq_len - context_asp_len)
                assert len(new_dep_tag) == len(context_asp_ids)
            pos_mask = [0] + pos_mask + [0] + [1] * len(term_tokens) + [0]
            pos_mask += paddings
            assert len(pos_mask) == len(context_asp_ids)

            context_asp_ids = np.asarray(context_asp_ids, dtype='int64')
            context_asp_seg_ids = np.asarray(context_asp_seg_ids, dtype='int64')
            context_asp_attention_mask = np.asarray(context_asp_attention_mask, dtype='int64')
            src_mask = np.asarray(src_mask, dtype='int64')
            aspect_mask = np.asarray(aspect_mask, dtype='int64')
            new_dep_tag = np.asarray(new_dep_tag, dtype='int64')
            pos_mask = np.asarray(pos_mask, dtype='int64')

            data = {
                'text_bert_indices': context_asp_ids,
                'bert_segments_ids': context_asp_seg_ids,
                'attention_mask': context_asp_attention_mask,
                'src_mask': src_mask,
                'aspect_mask': aspect_mask,
                'polarity': polarity,
                'deptag': new_dep_tag,
                'pos_mask': pos_mask
            }

            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
