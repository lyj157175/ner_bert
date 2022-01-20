import random
import numpy as np
import os

import torch
from pytorch_pretrained_bert import BertTokenizer



class DataLoader(object):

    def __init__(self, data_dir, bert_model_dir, params):
        self.data_dir = data_dir
        self.batch_size = params.batch_size
        self.max_len = params.max_len
        self.device = params.device
        self.seed = params.seed
        self.token_pad_idx = 0

        tags = self.load_tags()
        self.tag2idx = {tag: idx for idx, tag in enumerate(tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(tags)}
        params.tag2idx = self.tag2idx
        params.idx2tag = self.idx2tag
        self.tag_pad_idx = self.tag2idx['O']
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=True)


    def load_tags(self):
        tags = []
        file_path = os.path.join(self.data_dir, 'tags.txt')
        with open(file_path, 'r') as file:
            for tag in file:
                tags.append(tag.strip())
        return tags


    def load_sentences_tags(self, sentences_file, tags_file, d):
        sentences = []
        tags = []

        with open(sentences_file, 'r', encoding='utf-8') as file:
            for line in file:
                tokens = self.tokenizer.tokenize(line.strip())
                sentences.append(self.tokenizer.convert_tokens_to_ids(tokens))
        
        with open(tags_file, 'r') as file:
            for line in file:
                tag_seq = [self.tag2idx.get(tag) for tag in line.strip().split(' ')]
                tags.append(tag_seq)

        assert len(sentences) == len(tags)
        for i in range(len(sentences)):
            assert len(tags[i]) == len(sentences[i])

        d['data'] = sentences
        d['tags'] = tags
        d['size'] = len(sentences)


    def load_data(self, data_type):
        data = {}
        sentences_file_path = os.path.join(self.data_dir, data_type, 'sentences.txt')
        tags_path = os.path.join(self.data_dir, data_type, 'tags.txt')
        self.load_sentences_tags(sentences_file_path, tags_path, data)

        return data


    def data_iterator(self, data, shuffle=False):
        order = list(range(data['size']))
        if shuffle:
            random.seed(self.seed)
            random.shuffle(order)

        for i in range(data['size']//self.batch_size):
            sentences = [data['data'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]
            tags = [data['tags'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]

            batch_len = len(sentences)

            batch_max_len = max([len(s) for s in sentences])
            max_len = min(batch_max_len, self.max_len)

            batch_data = self.token_pad_idx * np.ones((batch_len, max_len))
            batch_tags = self.tag_pad_idx * np.ones((batch_len, max_len))

            for j in range(batch_len):
                cur_len = len(sentences[j])
                if cur_len <= max_len:
                    batch_data[j][:cur_len] = sentences[j]
                    batch_tags[j][:cur_len] = tags[j]
                else:
                    batch_data[j] = sentences[j][:max_len]
                    batch_tags[j] = tags[j][:max_len]

            batch_data = torch.tensor(batch_data, dtype=torch.long)
            batch_tags = torch.tensor(batch_tags, dtype=torch.long)
            batch_data, batch_tags = batch_data.to(self.device), batch_tags.to(self.device)
    
            yield batch_data, batch_tags

