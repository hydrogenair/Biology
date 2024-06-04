import os
from mindspore.dataset import GeneratorDataset
from transformers import BertTokenizer

# 假设你有一个FASTA文件 dna_sequences.fasta
def read_fasta(file_path):
    with open(file_path, 'r') as f:
        sequences = []
        seq = ""
        for line in f:
            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                    seq = ""
            else:
                seq += line.strip()
        if seq:
            sequences.append(seq)
    return sequences

# 使用BERT tokenizer对DNA序列进行编码
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class DNADataset:
    def __init__(self, sequences):
        self.sequences = sequences

    def __getitem__(self, index):
        sequence = self.sequences[index]
        encoded_sequence = tokenizer(sequence, padding='max_length', truncation=True, max_length=512)
        return encoded_sequence['input_ids'], encoded_sequence['attention_mask']

    def __len__(self):
        return len(self.sequences)

sequences = read_fasta('dna_sequences.fasta')
dataset = DNADataset(sequences)
train_dataset = GeneratorDataset(dataset, column_names=['input_ids', 'attention_mask'])
