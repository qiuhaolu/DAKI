import torch
import numpy as np 
from data import *
from model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AdamW, AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer, AlbertModel,AlbertForMaskedLM, AutoTokenizer, AutoModelForTokenClassification, AutoConfig




#torch.manual_seed(10)
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


use_fine_grained_rel = False
train_dataset = dataset_sgrouping(tokenizer = tokenizer, fine_grained = use_fine_grained_rel)

#exit()



#print(' '.join(cui2name[train_dataset[0]['head']]))
#print(train_dataset.__len__())

#print(train_dataset[0])
#exit()
collate_fn = lambda batch: batchfn_sgrouping(batch, device)  # Pass the device via a lambda
trainloader = DataLoader(train_dataset, shuffle=True, batch_size = 256, collate_fn = collate_fn, pin_memory = False)

model = SGroupingAdapterModel_pretrain(PTMwithAdapterModel = PTMwithUMLSadapter, fine_grained = use_fine_grained_rel).to(device)

optimizer = Adam(model.parameters(), lr = 1e-5) # 1e-5



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


    model.save_pretrained_adapter(path = 'pretrained_adaptermodel/albert_sgrouping'+'_ep'+str(epoch+1)+'.pt')
        

