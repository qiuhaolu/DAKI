from torch.utils.data import Dataset
import tqdm
import torch
import random
from itertools import islice
import numpy as np
import copy, pickle
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader

#from scispacy.linking import EntityLinker
#linker = EntityLinker(resolve_abbreviations=True, max_entities_per_mention = 1, threshold = 0.9, name="umls")

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results


class UMLSDataset(Dataset):
    def __init__(self, skiprate, tokenizer, rel_vocab, rel_fine_vocab, concepts, relations, concept_index=None, debug=False):
        self.tokenizer = tokenizer
        self.maxseqlength = 16
        self.relation = {}
        #f = open(rel_vocab)
        #for line in f:
        #    i, r, _ = line.strip().split()
        #    self.relation[r] = int(i)
        #num_rel = len(self.relation)
        self.use_fine_realtions = True
        if rel_fine_vocab:
            self.use_fine_realtions = True
            f = open(rel_fine_vocab)
            for line in f:
                i, r, _ = line.strip().split()
                #self.relation[r] = int(i) + num_rel
                self.relation[r] = int(i)
        #print(num_rel)
        #print(len(self.relation))
        #exit()
        self.concepts = {}
        self.conceptlist = []

        self.conceptlist_defi = []

        self.cui2text = {}
        #cnt = 0
        f = open(concepts)
        for line in f:
            CUI, name, name_seg = line.strip().split('\t')
            #if(CUI not in linker.kb.cui_to_entity):
            #    continue # if missing text, skip
            self.concepts[CUI] = name_seg.split()
            self.conceptlist.append(CUI)
            self.cui2text[CUI] = name_seg
            #print(self.concepts[CUI])
            #print(name)
            #print(name_seg)
            #exit()

            #self.cui2text[CUI] = linker.kb.cui_to_entity[CUI][1] + ' ' + linker.kb.cui_to_entity[CUI][4]
            #print(self.cui2text[CUI][0])
            #exit()
            #if(CUI in linker.kb.cui_to_entity and linker.kb.cui_to_entity[CUI][4] is not None):
            #    self.cui2text[CUI] = linker.kb.cui_to_entity[CUI]
            #    cnt += 1

        #print(cnt) #  ~10% missing linking entity 196187
        #exit()
        self.use_challenging_neg_samples = False
        if concept_index and not debug:
            self.use_challenging_neg_samples = True
            self.concept_index = pickle.load(open(concept_index, 'rb'))




        



        self.cui2defi = {}
        with open('/disk/luqh/UMLS/dataset/MRDEF.RRF') as f:
            for line in f:
                line = line.split('|')
                #cui = line[0]
                #defi = line[5]
                #print(cui)
                #print(defi)
                #exit()
                self.cui2defi[line[0]] = line[5]
                if(line[0] in self.concepts):
                    self.conceptlist_defi.append(line[0])

        #print(len(self.concepts)) #2983840
        #print(len(self.conceptlist_defi)) #333005  (if remove line 92, 383351)
        #exit()

        self.data = []
        f = open(relations)
        for i, line in enumerate(f):#islice(f, 100000):

            head, tail, rel, rel_fine = line.strip().split('\t')
            if(head not in self.cui2defi or tail not in self.cui2defi): # skip triples with no definitions for head or tail, resulting 6106155 triples (from 23029716)
                continue
            if(rel_fine == '\\N'):
                continue
            #print(head) #C1500000
            #exit()
            #if(head not in linker.kb.cui_to_entity or tail not in linker.kb.cui_to_entity):
            #    continue # skip triples with no text descriptions, resulting 21493111 triples (from 23029716)
            if(random.random() < skiprate):
                continue



            self.data.append((head, tail, rel, rel_fine))
            if debug and i == 100000: break
        #print(self.data[0:20])
        #print(len(self.data)) 
        hids_s = 0
        tids_s = 0
        #maxhids = 0
        #for i in range(len(self.data)):
        #    head, tail, rel, rel_fine = self.data[i]
        #    hdefi = self.cui2defi[head]
        #    tdefi = self.cui2defi[tail]
        #    hids = self.tokenizer(hdefi)['input_ids']
        #    tids = self.tokenizer(tdefi)['input_ids']
        #    hids_s += len(hids)
        #    tids_s += len(tids)
            #if(len(hids)>maxhids):
            #    maxhids = len(hids)
            #print(hdefi)
            #print(hids)
            #print(tdefi)
            #print(tids)
            #print(self.tokenizer.cls_token_id) #2

            #exit()
        #print(hids_s/len(self.data))
        #print(tids_s/len(self.data))
        #print(maxhids)
        #exit()

    def __len__(self):
        return len(self.data)

    def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
                tokens_a.pop()
            elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
                tokens_b.pop()
            elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
                tokens_c.pop()
            else:
                tokens_c.pop()

    def _sample_neg(self, CUI):
        if self.use_challenging_neg_samples and random.uniform(0, 1) < 0.6:
            name_seg = self.concepts[CUI]
            word = random.choice(name_seg)
            neg_CUI = self.conceptlist[random.choice(self.concept_index[word.lower()])]
            if neg_CUI != CUI:
                return neg_CUI
        #return random.choice(self.conceptlist)
        return random.choice(self.conceptlist_defi)

    def __getitem__(self, index):
        head, tail, rel, rel_fine = self.data[index]
        # sample corrupted head or tail
        if random.uniform(0, 1) < 0.5:
            head_neg = self._sample_neg(head)
            tail_neg = tail
        else:
            head_neg = head
            tail_neg = self._sample_neg(tail)

        if self.use_fine_realtions and rel_fine != '\\N':
            rel = rel_fine



        #transE version from xiao
        '''
        head = self.cui2text[head] + ' ' + self.cui2defi[head]
        tail = self.cui2text[tail] + ' ' + self.cui2defi[tail]
        head_neg = self.cui2text[head_neg] + ' ' + self.cui2defi[head_neg]
        tail_neg = self.cui2text[tail_neg] + ' ' + self.cui2defi[tail_neg]

        head_ids = self.tokenizer(head, padding = 'max_length', truncation=True, max_length = 64)['input_ids']
        tail_ids = self.tokenizer(tail, padding = 'max_length', truncation=True, max_length = 64)['input_ids']
        head_neg_ids = self.tokenizer(head_neg, padding = 'max_length', truncation=True, max_length = 64)['input_ids']
        tail_neg_ids = self.tokenizer(tail_neg, padding = 'max_length', truncation=True, max_length = 64)['input_ids']


        output = {"head": head_ids,
                  "rel": self.relation[rel],
                  "tail": tail_ids,
                  "head_neg": head_neg_ids,
                  "tail_neg": tail_neg_ids}
        return output
        '''


        #link prediction
        #print(rel)
        #print(rel_fine)
        rel = ' '.join(rel.split('_'))
        #print(rel)


        #with defi
        #head = self.cui2text[head] + ' ' + self.cui2defi[head]
        #tail = self.cui2text[tail] + ' ' + self.cui2defi[tail]
        #head_neg = self.cui2text[head_neg] + ' ' + self.cui2defi[head_neg]
        #tail_neg = self.cui2text[tail_neg] + ' ' + self.cui2defi[tail_neg]


        #without defi
        head = self.cui2text[head]
        tail = self.cui2text[tail]
        head_neg = self.cui2text[head_neg]
        tail_neg = self.cui2text[tail_neg]



        #triple_pos = head + ' ' + rel + ' ' + tail
        #triple_neg = head_neg + ' ' + rel + ' ' + tail_neg

        #tokens_rel = self.tokenizer.tokenize(rel)
        #print(self.tokenizer.tokenize('concept in subset'))
        #tokens_head = self.tokenizer.tokenize(head)
        #tokens_tail = self.tokenizer.tokenize(tail)
        #tokens_head_neg = self.tokenizer.tokenize(head_neg)
        #tokens_tail_neg = self.tokenizer.tokenize(tail_neg)

        triple_pos = head + ' ' + self.tokenizer.sep_token + ' ' + rel + ' ' + self.tokenizer.sep_token + ' ' + tail
        triple_neg = head_neg + ' ' + self.tokenizer.sep_token + ' ' + rel + ' ' + self.tokenizer.sep_token + ' ' + tail_neg
        triple_pos_tokenized = self.tokenizer(triple_pos, padding = 'max_length', truncation=True, max_length = self.maxseqlength)
        triple_neg_tokenized = self.tokenizer(triple_neg, padding = 'max_length', truncation=True, max_length = self.maxseqlength)

        #print(triples_pos_ids)
        #print(self.tokenizer.convert_ids_to_tokens(triples_pos_ids))
        print(triple_pos)

        #self._truncate_seq_triple(tokens_head, tokens_rel, tokens_tail, self.maxseqlength - 4)
        #self._truncate_seq_triple(tokens_head_neg, tokens_rel, tokens_tail_neg, self.maxseqlength - 4)
        #print(triple_pos)
        #print(triple_neg)
        #print(self.tokenizer.pad_token_id)
        #print(tokens_rel)
        #_truncate_seq_triple()

        output = {
            'triple_pos_ids':triple_pos_tokenized['input_ids'],
            'triple_pos_mask':triple_pos_tokenized['attention_mask'],
            'triple_neg_ids':triple_neg_tokenized['input_ids'],
            'triple_neg_mask':triple_neg_tokenized['attention_mask']
        }

        return output





    '''
    def concept_lookup(self, CUI):
        name_seg = self.concepts[CUI]
        tokens = name_seg[:10]      # limit name to 10 words
        tokens = [self.vocab.get(w.lower(), 1) for w in tokens]     # return 1 for UNK
        return tokens
    '''

class UMLSEntityPredictionDataset(Dataset):
    def __init__(self, vocab, rel_vocab, rel_fine_vocab, concepts, relations, debug=False):
        self.vocab = {}
        self.vocablist = []
        f = open(vocab)
        for line in f:
            i, w, _ = line.strip().split()
            self.vocab[w] = int(i)
            self.vocablist.append(w)
        self.relation = {}
        f = open(rel_vocab)
        for line in f:
            i, r, _ = line.strip().split()
            self.relation[r] = int(i)
        num_rel = len(self.relation)
        f = open(rel_fine_vocab)
        for line in f:
            i, r, _ = line.strip().split()
            self.relation[r] = int(i) + num_rel

        self.concepts = {}
        f = open(concepts)
        for line in f:
            CUI, name, name_seg = line.strip().split('\t')
            self.concepts[CUI] = name_seg.split()
        self.concepts_encode_cache = {}

        self.data = []
        f = open(relations)
        for i, line in enumerate(f): #islice(f, 3000):
            head, tail, rel, rel_fine, neg_tails = line.strip().split('\t')
            neg_tails = neg_tails.split()
            self.data.append((head, tail, rel, rel_fine, neg_tails))
            if debug and i == 100: break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        head, tail, rel, rel_fine, neg_tails = self.data[item]

        if rel_fine != '\\N':
            rel = rel_fine

        head, head_mask = stack_concepts([self.concept_lookup(head)])
        rel = [self.relation[rel]]
        tail, tail_mask = stack_concepts([self.concept_lookup(tail)])
        tail_neg, tail_neg_mask = stack_concepts([self.concept_lookup(t) for t in neg_tails])

        output = {"head": head,
                  "head_mask": head_mask,
                  "rel": rel,
                  "tail": tail,
                  "tail_mask": tail_mask,
                  "head_neg": head,
                  "head_neg_mask": head_mask,
                  "tail_neg": tail_neg,
                  "tail_neg_mask": tail_neg_mask}

        output = {key: torch.tensor(value) for key, value in output.items()}
        return output

    def concept_lookup(self, CUI):
        try:
            return self.concepts_encode_cache[CUI]
        except KeyError:
            name_seg = self.concepts[CUI]
            tokens = name_seg[:10]      # limit name to 10 words
            tokens = [self.vocab.get(w.lower(), 1) for w in tokens]     # return 1 for UNK
            self.concepts_encode_cache[CUI] = tokens
            return tokens

class UMLSEntityPredictionDatasetFull(Dataset):
    def __init__(self, vocab, rel_vocab, rel_fine_vocab, concepts, relations, debug=False):
        self.vocab = {}
        self.vocablist = []
        f = open(vocab)
        for line in f:
            i, w, _ = line.strip().split()
            self.vocab[w] = int(i)
            self.vocablist.append(w)
        self.relation = {}
        f = open(rel_vocab)
        for line in f:
            i, r, _ = line.strip().split()
            self.relation[r] = int(i)
        num_rel = len(self.relation)
        f = open(rel_fine_vocab)
        for line in f:
            i, r, _ = line.strip().split()
            self.relation[r] = int(i) + num_rel

        self.concepts = {}
        self.concept_i2c = []
        self.concept_c2i = {}
        f = open(concepts)
        for line in f:
            CUI, name, name_seg = line.strip().split('\t')
            self.concepts[CUI] = name_seg.split()
            self.concept_c2i[CUI] = len(self.concept_i2c)
            self.concept_i2c.append(CUI)

        self.data = []
        f = open(relations)
        for i, line in enumerate(f): #islice(f, 3000):
            head, tail, rel, rel_fine, excluded_neg_tails = line.strip().split('\t')
            excluded_neg_tails = excluded_neg_tails.split()
            self.data.append((head, tail, rel, rel_fine, excluded_neg_tails))
            if debug and i == 100: break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        head, tail, rel, rel_fine, excluded_neg_tails = self.data[item]

        if rel_fine != '\\N':
            rel = rel_fine

        head, head_mask = stack_concepts([self.concept_lookup(head)])
        rel = [self.relation[rel]]
        tail, tail_mask = stack_concepts([self.concept_lookup(tail)])
        excluded_neg_tails = [self.concept_c2i[c] for c in excluded_neg_tails]

        output = {"head": head,
                  "head_mask": head_mask,
                  "rel": rel,
                  "tail": tail,
                  "tail_mask": tail_mask,
                  "excluded_neg_tails": excluded_neg_tails}

        output = {key: torch.tensor(value) for key, value in output.items()}
        return output

    def concept_lookup(self, CUI):
        try:
            return self.concepts_encode_cache[CUI]
        except KeyError:
            name_seg = self.concepts[CUI]
            tokens = name_seg[:10]      # limit name to 10 words
            tokens = [self.vocab.get(w.lower(), 1) for w in tokens]     # return 1 for UNK
            self.concepts_encode_cache[CUI] = tokens
            return tokens

    def get_all_concepts(self):
        all_c, all_c_mask = stack_concepts([self.concept_lookup(c) for c in self.concept_i2c])
        output = {"all": all_c,
                  "all_mask": all_c_mask}
        return {key: torch.tensor(value) for key, value in output.items()}


class UMLSEntityPredictionDatasetv2(Dataset):
    def __init__(self, vocab, rel_vocab, rel_fine_vocab, concepts, relations):
        self.vocab = {}
        f = open(vocab)
        for line in f:
            i, w, _ = line.strip().split()
            self.vocab[w] = int(i)
        self.relation = {}
        f = open(rel_vocab)
        for line in f:
            i, r, _ = line.strip().split()
            self.relation[r] = int(i)
        num_rel = len(self.relation)
        if rel_fine_vocab:
            self.use_fine_realtions = True
            f = open(rel_fine_vocab)
            for line in f:
                i, r, _ = line.strip().split()
                self.relation[r] = int(i) + num_rel
        else:
            self.use_fine_realtions = False

        self.concepts = {}
        f = open(concepts)
        for line in f:
            CUI, name = line.strip().split('\t')
            self.concepts[CUI] = name
        self.concepts_encode_cache = {}

        self.data = []
        f = open(relations)
        for line in f: #islice(f, 30000):
            head, tail, rel, rel_fine = line.strip().split('\t')
            self.data.append((head, tail, rel, rel_fine))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        head, tail, rel, rel_fine = self.data[item]

        if self.use_fine_realtions and rel_fine != '\\N':
            rel = rel_fine

        output = {"head": self.concept_lookup(head),
                  "rel": self.relation[rel],
                  "tail": self.concept_lookup(tail),
                  "head_neg": [0],  # dummy neg here
                  "tail_neg": [0]}

        return output

    def get_all_concepts(self):
        all_c, all_c_mask = stack_concepts([self.concept_lookup(c) for c in self.concepts])
        output = {"all": all_c,
                  "all_mask": all_c_mask}
        return {key: torch.tensor(value) for key, value in output.items()}

    def concept_lookup(self, CUI):
        try:
            return self.concepts_encode_cache[CUI]
        except KeyError:
            name = self.concepts[CUI]
            tokens = name.split()[:10]      # limit name to 10 words
            tokens = [self.vocab.get(w.lower(), 1) for w in tokens]     # return 1 for UNK
            self.concepts_encode_cache[CUI] = tokens
            return tokens


def stack_concepts(cs):
    batch_size = len(cs)
    max_length = max(len(s) for s in cs)
    mask = np.zeros([batch_size, max_length], dtype=np.float32)
    css = np.zeros([batch_size, max_length], dtype=np.int64)
    #mask = [[1.0]*len(c) + [0]*(max_length-len(c)) for c in cs]     # generate mask first (because it uses the true length of cs)
    #cs = [c + [0]*(max_length-len(c)) for c in cs]
    for i,s in enumerate(cs):
        mask[i, :len(s)] = 1.
        css[i, :len(s)] = s
    return css, mask


def batchify(data):
    head, head_mask = stack_concepts([d['head'] for d in data])
    tail, tail_mask = stack_concepts([d['tail'] for d in data])
    head_neg, head_neg_mask = stack_concepts([d['head_neg'] for d in data])
    tail_neg, tail_neg_mask = stack_concepts([d['tail_neg'] for d in data])
    rel = [d['rel'] for d in data]

    output = {"head": head,
              "head_mask": head_mask,
              "rel": rel,
              "tail": tail,
              "tail_mask": tail_mask,
              "head_neg": head_neg,
              "head_neg_mask": head_neg_mask,
              "tail_neg": tail_neg,
              "tail_neg_mask": tail_neg_mask}

    return {key: torch.tensor(value) for key, value in output.items()}

'''
def batchfn(data):
    head_idss = [d['head'] for d in data]
    tail_idss = [d['tail'] for d in data]
    head_neg_idss = [d['head_neg'] for d in data]
    tail_neg_idss = [d['tail_neg'] for d in data]
    rel_idss = [d['rel'] for d in data]

    head_idss = torch.tensor(head_idss).cuda()
    tail_idss = torch.tensor(tail_idss).cuda()
    head_neg_idss = torch.tensor(head_neg_idss).cuda()
    tail_neg_idss = torch.tensor(tail_neg_idss).cuda()
    rel_idss = torch.tensor(rel_idss).cuda()

    output = {"head_idss": head_idss,
                  "rel_idss": rel_idss,
                  "tail_idss": tail_idss,
                  "head_neg_idss": head_neg_idss,
                  "tail_neg_idss": tail_neg_idss}

    return output
'''

def batchfn(data):
    triple_pos_ids_s = [d['triple_pos_ids'] for d in data]
    triple_pos_mask_s = [d['triple_pos_mask'] for d in data]
    triple_neg_ids_s = [d['triple_neg_ids'] for d in data]
    triple_neg_mask_s = [d['triple_neg_mask'] for d in data]

    triple_pos_ids_s = torch.tensor(triple_pos_ids_s).cuda()
    triple_pos_mask_s = torch.tensor(triple_pos_mask_s).cuda()
    triple_neg_ids_s = torch.tensor(triple_neg_ids_s).cuda()
    triple_neg_mask_s = torch.tensor(triple_neg_mask_s).cuda()

    output = {
        'triple_pos_ids_s': triple_pos_ids_s,
        'triple_pos_mask_s': triple_pos_mask_s,
        'triple_neg_ids_s': triple_neg_ids_s,
        'triple_neg_mask_s': triple_neg_mask_s
    }

    return output





###disease part
class dataset_disease(Dataset):
    def __init__(self, file, tokenizer):
        super().__init__()
        self.data = []
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #self.model = BertForMaskedLM.from_pretrained('/disk/luqh/pretrained_lm/bert-base-uncased',return_dict=True)
        self.tokenizer = tokenizer

        #self.tokenizer.add_tokens('[blank]') 
        #self.model.resize_token_embeddings(len(self.tokenizer))

        #inputs = self.tokenizer("The capital of France is [MASK] [MASK].", return_tensors="pt")
        #labels = self.tokenizer("The capital of France is Paris a.", return_tensors="pt")["input_ids"]
        #print(labels)
        #exit()
        #outputs = self.model(**inputs, labels=labels)
        #loss = outputs.loss
        #logits = outputs.logits

        #print(loss)
        #print(logits)

        with open(file, 'r') as f:
            rawdata = f.readlines()
        cnt = 0
        lengths = 0
        #print(len(rawdata))
        for line in rawdata:
            #print(len(line.split('\t')))
            values = line.split('\t') # all 4 columns, [disease, aspect, auxiliary sentence, passage]
            disease_name = values[0]
            auxed_passage = values[2] + ' ' + values[3].strip()

            #cnt+=1
            #if(cnt>1):
            #   exit()
            #print(disease_name)
            #print(auxed_passage)
            #token_auxed_passage = self.tokenizer(auxed_passage, padding = 'max_length', truncation=True, max_length = 256)['input_ids']
            #print(len(token_auxed_passage))
            #lengths+=len(token_auxed_passage)
            
            token_disease = self.tokenizer(disease_name)['input_ids']
            token_auxed_passage = self.tokenizer(auxed_passage, padding = 'max_length', truncation=True, max_length = 256)['input_ids']
            token_auxed_passage_ori = token_auxed_passage.copy()

            sublists = find_sub_list(token_disease[1:len(token_disease)-1],token_auxed_passage)

            
            for lst in sublists:
                lst_s = lst[0]
                lst_e = lst[1]
                for i in range(lst_s,lst_e+1):
                    if(random.random() < 0.75):
                        token_auxed_passage[i] = self.tokenizer.mask_token_id


            #print(token_auxed_passage)
            #print(token_auxed_passage_ori)
            #print(self.tokenizer.mask_token_id)
            #exit()
            self.data.append((token_auxed_passage, token_auxed_passage_ori))
            

            #self.data.append((disease_name,auxed_passage))
            #print(token_auxed_passage)
            #print(token_auxed_passage_ori)
            #print(len(token_auxed_passage))
            #print(len(token_auxed_passage_ori))


            #tokenized_auxed_passage = self.tokenizer(auxed_passage)['input_ids']
            #print(len(tokenized_auxed_passage))#503
            #print(len(self.tokenizer.convert_ids_to_tokens(tokenized_auxed_passage)))#503
            #print(len(self.tokenizer.decode(tokenized_auxed_passage)))#1921

            #print(self.model())
        #print(len(self.data))

        #avg_length = lengths/len(self.data)
        #print(avg_length)



    def __getitem__(self, index):
        token_auxed_passage, token_auxed_passage_ori = self.data[index]

        return {'token_auxed_passage':token_auxed_passage,
                'token_auxed_passage_ori':token_auxed_passage_ori
                }

    def __len__(self):
        return len(self.data)



def batchfn_disease(data):
    token_auxed_passages = [d['token_auxed_passage'] for d in data]
    token_auxed_passages_ori = [d['token_auxed_passage_ori'] for d in data]
    token_auxed_passages = torch.tensor(token_auxed_passages).cuda()
    token_auxed_passages_ori = torch.tensor(token_auxed_passages_ori).cuda()
    return {'token_auxed_passages':token_auxed_passages,
            'token_auxed_passages_ori':token_auxed_passages_ori
    }












###semantic grouping part
class dataset_sgrouping(Dataset):
    def __init__(self, tokenizer, fine_grained):
        super().__init__()
        self.data = []

        self.tokenizer = tokenizer


        cnt = 0
        lengths = 0


        self.cui2text = {}
        with open('/disk/luqh/UMLS/dataset/concepts.txt') as f:
            for line in f:
                CUI, name, name_seg = line.strip().split('\t')
                #if(CUI not in linker.kb.cui_to_entity):
                #    continue # if missing text, skip
                #self.concepts[CUI] = name_seg.split()
                #self.conceptlist.append(CUI)
                self.cui2text[CUI] = name_seg






        self.sgroup2id = {'ACTI':0,'ANAT':1,'CHEM':2,'CONC':3,'DEVI':4,'DISO':5,'GENE':6,'GEOG':7,'LIVB':8,'OBJC':9,'OCCU':10,'ORGA':11,'PHEN':12,'PHYS':13,'PROC':14}
        self.tui2tid = {}
        temptid = 0
        self.tui2sgroupid = {}
        with open('/disk/luqh/UMLS/dataset/SemGroups.txt') as f:
            for line in f:
                line = line.split('|')
                #cui = line[0]
                #defi = line[5]
                #print(cui)
                #print(defi)
                #exit()
                self.tui2sgroupid[line[2]] = self.sgroup2id[line[0]]
                self.tui2tid[line[2]] = temptid
                temptid += 1
                #if(line[0] in self.concepts):
                #    self.conceptlist_defi.append(line[0])
        #print(self.tui2sgroupid)
        #print(len(self.tui2sgroupid)) #127
        #print(self.tui2tid)
        #exit()


        self.cui2sgroupid = {}
        self.cui2styid = {}
        with open('/disk/luqh/UMLS/dataset/MRSTY.RRF') as f:
            for line in f:
                line = line.split('|')
                #cui = line[0]
                #defi = line[5]
                #print(cui)
                #print(defi)
                #exit()
                self.cui2sgroupid[line[0]] = self.tui2sgroupid[line[1]]
                self.cui2styid[line[0]] = self.tui2tid[line[1]]
                #if(line[0] in self.concepts):
                #    self.conceptlist_defi.append(line[0])
        #cnt = 0
        #print(len(self.cui2sgroupid))
        #for cui in self.cui2sgroupid:
        #    print(self.cui2sgroupid[cui])
        #    cnt+=1
        #    if(cnt>10):
        #        exit()
        #exit()


        #self.cui2defi = {}
        with open('/disk/luqh/UMLS/dataset/MRDEF.RRF') as f:
            for line in f:
                line = line.split('|')

                #self.cui2defi[line[0]] = line[5]
                if(fine_grained == True):
                    if(line[0] in self.cui2styid and line[0] in self.cui2text):
                        label = self.cui2styid[line[0]]
                        text = self.cui2text[line[0]] + ' ' + line[5]
                        tokens_defi = self.tokenizer(text, padding = 'max_length', truncation=True, max_length = 128)
                        #print(tokens_defi['input_ids'])
                        #print(tokens_defi['attention_mask'])
                        #exit()
                        #self.data.append((line[0], tokens_defi, label))
                        self.data.append((tokens_defi, label))
                else:
                    if(line[0] in self.cui2sgroupid and line[0] in self.cui2text):
                        label = self.cui2sgroupid[line[0]]
                        text = self.cui2text[line[0]] + ' ' + line[5]
                        tokens_defi = self.tokenizer(text, padding = 'max_length', truncation=True, max_length = 128)
                        #print(tokens_defi['input_ids'])
                        #print(tokens_defi['attention_mask'])
                        #exit()
                        #self.data.append((line[0], tokens_defi, label))
                        self.data.append((tokens_defi, label))

        #print(len(self.data))
        #print(self.data[10])
        #exit()








    def __getitem__(self, index):
        tokens_defi, label = self.data[index]

        return {'defi_input_ids':tokens_defi['input_ids'],
                'defi_mask':tokens_defi['attention_mask'],
                'label':label
                }

    def __len__(self):
        return len(self.data)



def batchfn_sgrouping(data):
    defi_input_ids_batch = [d['defi_input_ids'] for d in data]
    defi_mask_batch = [d['defi_mask'] for d in data]
    defi_label_batch = [d['label'] for d in data]

    defi_input_ids_batch = torch.tensor(defi_input_ids_batch).cuda()
    defi_mask_batch = torch.tensor(defi_mask_batch).cuda()
    defi_label_batch = torch.tensor(defi_label_batch).cuda()

    return {
    'defi_input_ids_batch':defi_input_ids_batch,
    'defi_mask_batch':defi_mask_batch,
    'defi_label_batch':defi_label_batch
    }




