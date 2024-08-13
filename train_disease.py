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



diseaseBERT_data_url = 'extractedQuestionAnswers_total_removeNoisy_maskedLM.txt'




ptm_model = AlbertModel.from_pretrained('albert-xxlarge-v2',output_hidden_states=True)
tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')




PTMwithDiseaseAdapter = PTMwithAdapterModel(device = device, model = ptm_model, tokenizer = tokenizer, adaptersize = 128, petrained_adaptermodel_path = None,
                            freeze_encoder = True, freeze_adapter = False, fusion_mode = 'add')



train_dataset = dataset_disease(diseaseBERT_data_url, tokenizer)
print(train_dataset.__len__())

#print(train_dataset[0])
#exit()

collate_fn = lambda batch: batchfn_disease(batch, device)  # Pass the device via a lambda
trainloader = DataLoader(train_dataset, batch_size = 16, collate_fn = collate_fn, pin_memory = False)
model = diseaseAdapterModel_pretrain(PTMwithDiseaseAdapter).to(device)
optimizer = Adam(model.parameters(), lr = 2e-4)

model.train()


for epoch in range(10):
    tqdm_trainloader = tqdm(trainloader)

    for batch in tqdm_trainloader:

        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        tqdm_trainloader.set_postfix(loss = loss.item())

    model.save_pretrained_adapter(path = 'pretrained_adaptermodel/albert_disease_adaptermodel'+'_ep'+str(epoch+1)+'.pt')

