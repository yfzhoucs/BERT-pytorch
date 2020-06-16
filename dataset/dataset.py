from torch.utils.data import Dataset
import tqdm
import torch
import random


class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True, multi_segment=False, sep=False):
        self.vocab = vocab
        self.seq_len = seq_len
        self.multi_segment = multi_segment
        self.sep = sep

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [[line[:-1].strip(), '']
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1, t2, is_next_label = self.random_sent(item)
        t1_raw, t1_random, t1_label, t1_segment_labels = self.random_word(t1, self.sep)

        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t1_raw = [self.vocab.sos_index] + t1_raw + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]

        if not self.multi_segment:
            segment_label = [1 for _ in range(len(t1))]
        else:
            segment_label = [5] + t1_segment_labels + [6]
        bert_input = (t1)[:self.seq_len]
        bert_label = (t1_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding)
        bert_label.extend(padding)
        segment_label.extend(padding)
        t1_raw.extend(padding)


        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label,
                  "t1_raw": t1_raw}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence, sep):
        raw_tokens = sentence.split()
        tokens = sentence.split()
        output_label = []
        segment_label = []

        if not sep:
            for i, token in enumerate(tokens):
                if token.isdigit():
                    segment_label.append(int(token) // 100 + 1)
                raw_tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                prob = random.random()
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = self.vocab.mask_index

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.randrange(len(self.vocab))

                    # 10% randomly change token to current token
                    else:
                        tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                    output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                    output_label.append(0)
        else:
            tokens_wo_sep = sentence.split()
            raw_tokens = []
            input_tokens = []
            pointer = 1
            tokens = []
            for token in tokens_wo_sep:
                if int(token) // 100 + 1 > pointer:
                    tokens.append('<sep>')
                    tokens.append(token)
                    pointer = int(token) // 100 + 1
                else:
                    tokens.append(token)

            for i, token in enumerate(tokens):
                if token.isdigit():
                    segment_label.append(int(token) // 100 + 1)
                    raw_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                    prob = random.random()
                    if prob < 0.15:
                        prob /= 0.15

                        # 80% randomly change token to mask token
                        if prob < 0.8:
                            input_tokens.append(self.vocab.mask_index)

                        # 10% randomly change token to random token
                        elif prob < 0.9:
                            input_tokens.append(random.randrange(len(self.vocab)))

                        # 10% randomly change token to current token
                        else:
                            input_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))

                        output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

                    else:
                        input_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                        output_label.append(0)
                else:
                    raw_tokens.append(self.vocab.eos_index)
                    input_tokens.append(self.vocab.eos_index)
                    output_label.append(0)
                    segment_label.append(5)
                tokens = input_tokens

        return raw_tokens, tokens, output_label, segment_label

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2

    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]


if __name__ == '__main__':
    import argparse
    from  vocab import WordVocab
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()        
    parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset for train bert")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-s", "--seq_len", type=int, default=20, help="maximum sequence len")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="number of batch_size")  
    parser.add_argument("--multi_segment", type=bool, default=False, help="whether to use multiple segment_labels for entity types")  
    parser.add_argument("--sep_label", type=bool, default=False, help="whether to insert <sep>")  
    args = parser.parse_args()

    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset, vocab, seq_len=args.seq_len,
                                corpus_lines=args.corpus_lines, on_memory=args.on_memory, multi_segment=args.multi_segment, sep=args.sep_label)

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    for x in train_data_loader:
        print(x)
        input()