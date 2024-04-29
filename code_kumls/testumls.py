import torch
import numpy as np 
from data import *
from model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AdamW, AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer, AlbertModel,AlbertForMaskedLM, AutoTokenizer, AutoModelForTokenClassification, AutoConfig



rel_vocab_file = '/disk/luqh/UMLS/dataset/rel_vocab.txt'
rel_file_vocab_file = '/disk/luqh/UMLS/dataset/rel_fine_vocab.txt'
concepts_file = '/disk/luqh/UMLS/dataset/concepts.txt'
#definition_file = '/disk/luqh/UMLS/dataset/MRDEF.RRF'
train_file = '/disk/luqh/UMLS/dataset/train.txt'
#train_file = '/disk/luqh/UMLS/dataset/valid.txt'
#train_file = '/disk/luqh/UMLS/dataset/relations.txt'


tokenizer = AlbertTokenizer.from_pretrained('/disk/luqh/pretrained_lm/albert-xxlarge-v2')

train_dataset = UMLSDataset(tokenizer = tokenizer, skiprate = 0, rel_vocab = rel_vocab_file, rel_fine_vocab = rel_file_vocab_file, concepts = concepts_file, relations = train_file, concept_index=None, debug=False)
cui2name = train_dataset.concepts

#print(' '.join(cui2name[train_dataset[0]['head']]))
#print(train_dataset.__len__())
print(train_dataset[0:10])
exit()

#trainloader = DataLoader(train_dataset, shuffle=True, batch_size = 256, collate_fn = batchfn, pin_memory = False)

