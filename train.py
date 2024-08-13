import torch
import numpy as np 
from data import *
from model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AdamW, AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer, AlbertModel,AlbertForMaskedLM, AutoTokenizer, AutoModelForTokenClassification, AutoConfig



#model = BertModel.from_pretrained('/disk/luqh/adapter/pretrained_adapters/umlstriple_old/')
#model.save_adapter('/disk/luqh/adapter/pretrained_adapters/umlstriple_old/', 'umlstriple')

#exit()


#torch.manual_seed(10)
device = torch.device("mps")
#torch.cuda.set_device(device)


'''
#test modelwithadapter
tokenizer = BertTokenizer.from_pretrained('/disk/luqh/pretrained_lm/bert-base-uncased')
model = PTMwithAdapterModel(device = device, petrained_adaptermodel_path = './pretrained_adaptermodel/adaptermodel1.pt')
model.to(device)

input_ids = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)]).to(device)  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
with torch.no_grad():
    last_hidden_states = model(input_ids)[0]  # Models outputs are tuples

# The last_hidden_state can be used as input for downstream tasks
#print('last_hidden_states:',last_hidden_states)
print('here')
#model.save_adaptermodel(path = './pretrained_adaptermodel/adaptermodel.pt')
#model.fac_adapter.save_adaptermodel(path = './pretrained_adaptermodel/adaptermodel1.pt')
exit()
'''





rel_vocab_file = 'UMLS/dataset/rel_vocab.txt'
rel_file_vocab_file = 'UMLS/dataset/rel_fine_vocab.txt'
concepts_file = 'UMLS/dataset/concepts.txt'
train_file = 'UMLS/dataset/train.txt'

#tokenizer = BertTokenizer.from_pretrained('/disk/luqh/pretrained_lm/bert-base-uncased')


ptm_model = AlbertModel.from_pretrained('albert-xxlarge-v2',output_hidden_states=True)
tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')



PTMwithUMLSadapter = PTMwithAdapterModel(device = device, model = ptm_model, tokenizer = tokenizer, adaptersize = 128, petrained_adaptermodel_path = None,
                            freeze_encoder = True, freeze_adapter = False, fusion_mode = 'add')


train_dataset = UMLSDataset(tokenizer = tokenizer, skiprate = 0.0, rel_vocab = rel_vocab_file, rel_fine_vocab = rel_file_vocab_file, concepts = concepts_file, relations = train_file, concept_index=None, debug=False)
cui2name = train_dataset.concepts

#print(' '.join(cui2name[train_dataset[0]['head']]))
print(train_dataset.__len__())

#print(train_dataset[0])
#exit()
collate_fn = lambda batch: batchfn(batch, device)  # Pass the device via a lambda
trainloader = DataLoader(train_dataset, shuffle=True, batch_size = 256, collate_fn = collate_fn, pin_memory = False)

#model = UMLSAdapterModel_pretrain(relation_size = len(train_dataset.relation), PTMwithAdapterModel = PTMwithUMLSadapter).to(device)
model = UMLSAdapterModel_LP_pretrain(PTMwithAdapterModel = PTMwithUMLSadapter).to(device)

optimizer = Adam(model.parameters(), lr = 1e-6) # ori 1e-5

#exit()

model.train()


for epoch in range(10):
    tqdm_trainloader = tqdm(trainloader)
    #print(epoch)
    for batch in tqdm_trainloader:
        #print(len(batch))
        #print(len(batch['head_idss']))
        #print(len(batch['rel_idss']))
        #print(batch['rel_idss'])
        #batch = tuple(t.to(device) for t in batch)

        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        tqdm_trainloader.set_postfix(loss = loss.item())
        exit()

    #model.save_pretrained_adapter(path = '/disk/luqh/adapter/code_kumls/pretrained_adaptermodel/albert_umlswithdefi_adaptermodel_skiprate0.9'+'_ep'+str(epoch+1)+'.pt')
    #model.save_pretrained_adapter(path = '/disk/luqh/adapter/code_kumls/pretrained_adaptermodel/albert_umlswithdefi_LP_skiprate0.7'+'_ep'+str(epoch+1)+'.pt')
    model.save_pretrained_adapter(path = 'pretrained_adaptermodel/albert_umlswithoutdefi_LP'+'_ep'+str(epoch+1)+'.pt')
        
        #exit()

