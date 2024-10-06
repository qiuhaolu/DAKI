import torch
import numpy as np 
from data import *
from model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AdamW, AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer, AlbertModel,AlbertForMaskedLM, AutoTokenizer, AutoModelForTokenClassification, AutoConfig




device = torch.device("mps")
#torch.cuda.set_device(device)




rel_vocab_file = 'UMLS/dataset/rel_vocab.txt'
rel_file_vocab_file = 'UMLS/dataset/rel_fine_vocab.txt'
concepts_file = 'UMLS/dataset/concepts.txt'
train_file = 'UMLS/dataset/train.txt'


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

model = UMLSAdapterModel_LP_pretrain(PTMwithAdapterModel = PTMwithUMLSadapter).to(device)

optimizer = Adam(model.parameters(), lr = 1e-6) # ori 1e-5


model.train()


for epoch in range(10):
    tqdm_trainloader = tqdm(trainloader)
    #print(epoch)
    for batch in tqdm_trainloader:

        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        tqdm_trainloader.set_postfix(loss = loss.item())
        exit()


    model.save_pretrained_adapter(path = 'pretrained_adaptermodel/albert_umlswithoutdefi_LP'+'_ep'+str(epoch+1)+'.pt')
        
  