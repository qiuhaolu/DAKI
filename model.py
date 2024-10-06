import torch.nn as nn
import torch
import pickle
from math import floor
import os
from transformers import BertTokenizer, BertModel, AdamW, AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer, AlbertModel,AlbertForMaskedLM, AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from transformers.models.albert.modeling_albert import AlbertMLMHead

from torch.nn import functional as F
from torch.optim import Adam

class Adapter(nn.Module):
    def __init__(self, hiddensize, adaptersize):
        super(Adapter, self).__init__()

        self.adapter_size = adaptersize #default 128
        self.project_hidden_size = hiddensize #bert-base 768 albert-xxlarge 4096
        self.down_project = nn.Linear(self.project_hidden_size,self.adapter_size)
        self.up_project = nn.Linear(self.adapter_size, self.project_hidden_size)

    def forward(self, hidden_states):
        down_projected = F.leaky_relu(self.down_project(hidden_states))
        up_projected = self.up_project(down_projected)

        return hidden_states + up_projected

class AdapterModel(nn.Module):
    def __init__(self,device, hiddensize, adaptersize):
        super().__init__()
        self.device = device
        self.adapter_list = [0,5,11]
        #self.adapter_list = [1,6,12]
        #self.adapter_list = [0,1,3,5,7,9,10,11,12]
        self.adapter_num = len(self.adapter_list)
        self.adapters = nn.ModuleList([Adapter(hiddensize, adaptersize) for _ in range(self.adapter_num)])

    def forward(self, pretrained_model_outputs):
        outputs = pretrained_model_outputs
        sequence_output = outputs[0]
        hidden_states = outputs[2]


        num = len(hidden_states)
        hidden_states_last = torch.zeros(sequence_output.size()).to(self.device)
        #print(self.device)

        adapter_hidden_states = []
        adapter_hidden_states_count = 0
        for i, adapter_module in enumerate(self.adapters):
            fusion_state = hidden_states[self.adapter_list[i]] + hidden_states_last
            hidden_states_last = adapter_module(fusion_state)
            adapter_hidden_states.append(hidden_states_last)


        adapter_hidden_states_tensor = torch.stack(adapter_hidden_states, dim = 1)



        outputs = hidden_states_last

        return outputs, adapter_hidden_states_tensor  # (loss), logits, (hidden_states), (attentions)

    def save_adaptermodel(self, path):

        torch.save(self.state_dict(), path)
        print('done saving adaptermodel')


class ControllerModel(nn.Module):
    def __init__(self,device, hiddensize, adaptersize):
        super().__init__()
        self.device = device
        #self.adapter_list = [0,1,2,3,4,5,6,7,8,9,10,11,12]  # controller at every layer
        self.adapter_list = [0,5,11]
        self.adapter_num = len(self.adapter_list)
        self.adapters = nn.ModuleList([Adapter(hiddensize, adaptersize) for _ in range(self.adapter_num)])

    def forward(self, pretrained_model_outputs):
        outputs = pretrained_model_outputs

        sequence_output = outputs[0]    
        hidden_states = outputs[2]


        num = len(hidden_states)
        hidden_states_last = torch.zeros(sequence_output.size()).to(self.device)  # task context vec: initialize with 0s #[8,128,4096]
 
        adapter_hidden_states = []
        adapter_hidden_states_count = 0
        for i, adapter_module in enumerate(self.adapters):
            fusion_state = hidden_states[self.adapter_list[i]] + hidden_states_last
            hidden_states_last = adapter_module(fusion_state)
            adapter_hidden_states.append(hidden_states_last)

        adapter_hidden_states_tensor = torch.stack(adapter_hidden_states, dim = 1)



        outputs = hidden_states_last

        return outputs, adapter_hidden_states_tensor  # (loss), logits, (hidden_states), (attentions)







class PTMwithAdapterModel(nn.Module):

    def __init__(self, device, model, tokenizer, adaptersize = 128, petrained_adaptermodel_path = None, freeze_encoder = True, freeze_adapter = False, fusion_mode = 'add'):
        super().__init__()

        self.device = device
        self.fusion_mode = fusion_mode
        self.freeze_encoder = freeze_encoder
        self.freeze_adapter = freeze_adapter

        self.tokenizer = tokenizer
        self.model = model

        if self.freeze_encoder:
            for p in self.model.parameters():
                p.requires_grad = False

        if(petrained_adaptermodel_path == None):
            self.fac_adapter = AdapterModel(self.device, self.model.config.hidden_size, adaptersize)
        else:
            self.fac_adapter = self.load_adaptermodel(petrained_adaptermodel_path)
            self.fac_adapter.device = self.device
 

        if self.freeze_adapter:
            for p in self.fac_adapter.parameters():
                p.requires_grad = False

 



    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask)



        pretrained_model_last_hidden_states = outputs[0]  # original bert/roberta output


        fac_adapter_outputs, _ = self.fac_adapter(outputs)

        task_features = pretrained_model_last_hidden_states

        if self.fusion_mode == 'add':
            task_features = task_features + fac_adapter_outputs

        return (task_features,)

    def save_adaptermodel(self, path):
        self.fac_adapter.save_adaptermodel(path)

    def load_adaptermodel(self, path):
        adapterm = AdapterModel(device = self.device, hiddensize=self.model.config.hidden_size, adaptersize=adaptersize).to(self.device)
        adapterm.load_state_dict(torch.load(path, map_location = self.device))
        return adapterm



class UMLSAdapterModel_pretrain(nn.Module):

    def __init__(self, relation_size, PTMwithAdapterModel):

        super().__init__()
        self.hidden = PTMwithAdapterModel.model.config.hidden_size
        self.p_norm = 2
        self.margin = 0.5

        self.encoder = PTMwithAdapterModel
        self.tokenizer = PTMwithAdapterModel.tokenizer



        self.relation_embedding = nn.Embedding(num_embeddings=relation_size, embedding_dim=self.hidden)
        # initialization
        nn.init.xavier_normal_(self.relation_embedding.weight)

        self.criterion = nn.MarginRankingLoss(self.margin)





    def forward(self, input):
        head_idss = input['head_idss']
        tail_idss = input['tail_idss']
        head_neg_idss = input['head_neg_idss']
        tail_neg_idss = input['tail_neg_idss']
        rel_idss = input['rel_idss']

        #print(len(rel_idss))
        batch_size = len(rel_idss)
        rel_emb = self.relation_embedding(rel_idss)
        #print(rel_emb.shape) #[batch_size, relation_size]
        #print(self.encoder(head_idss))
        head_emb = torch.mean(self.encoder(head_idss)[0], keepdim = False, dim = 1)  # or use CLS
        #print(head_emb.shape) #[batch_size, bert_size]
        tail_emb = torch.mean(self.encoder(tail_idss)[0], keepdim = False, dim = 1)
        head_neg_emb = torch.mean(self.encoder(head_neg_idss)[0], keepdim = False, dim = 1)
        tail_neg_emb = torch.mean(self.encoder(tail_neg_idss)[0], keepdim = False, dim = 1) 

        p_score = torch.norm(head_emb + rel_emb - tail_emb, self.p_norm, dim=-1)
        n_score = torch.norm(head_neg_emb + rel_emb - tail_neg_emb, self.p_norm, dim=-1)



        if self.training:
            loss = loss = self.criterion(p_score, n_score, torch.tensor([-1.0]*batch_size).to(self.encoder.device)) ## ori cuda()

            return loss
        else:#eval
            return 0

    def save_pretrained_adapter(self, path):
        self.encoder.save_adaptermodel(path)








class UMLSAdapterModel_LP_pretrain(nn.Module):

    def __init__(self, PTMwithAdapterModel):

        super().__init__()
        self.hidden = PTMwithAdapterModel.model.config.hidden_size

        self.encoder = PTMwithAdapterModel
        self.tokenizer = PTMwithAdapterModel.tokenizer

        self.linear = nn.Linear(self.hidden, 2)

        self.criterion = nn.CrossEntropyLoss()





    def forward(self, input):

        triple_pos_ids_s = input['triple_pos_ids_s']
        triple_pos_mask_s = input['triple_pos_mask_s']
        triple_neg_ids_s = input['triple_neg_ids_s']
        triple_neg_mask_s = input['triple_neg_mask_s']

       

        triple_pos_emb = self.encoder(input_ids = triple_pos_ids_s, attention_mask = triple_pos_mask_s)[0] # 0 only one element in tuple output
        triple_neg_emb = self.encoder(input_ids = triple_neg_ids_s, attention_mask = triple_neg_mask_s)[0]

  
        triple_pos_neg_cls = torch.cat([triple_pos_emb[:,0,:],triple_neg_emb[:,0,:]])
        label = torch.tensor([1] * len(triple_pos_ids_s) + [0] * len(triple_neg_ids_s)).to(self.encoder.device) ## ori cuda()

        output = self.linear(triple_pos_neg_cls)
 


        if self.training:
            loss = self.criterion(output, label)

            return loss
        else:#eval
            return 0

    def save_pretrained_adapter(self, path):
        self.encoder.save_adaptermodel(path)
















class diseaseAdapterModel_pretrain(nn.Module):

    def __init__(self, PTMwithAdapterModel):

        super().__init__()

        self.encoder = PTMwithAdapterModel
        self.tokenizer = PTMwithAdapterModel.tokenizer

        self.predictions = AlbertMLMHead(PTMwithAdapterModel.model.config)




    def forward(self, input):
        token_auxed_passages = input['token_auxed_passages']
        token_auxed_passages_ori = input['token_auxed_passages_ori']

        outputs = self.encoder(token_auxed_passages)
        sequence_outputs = outputs[0]
        prediction_scores = self.predictions(sequence_outputs)


        loss_fct = nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.encoder.model.config.vocab_size), token_auxed_passages_ori.view(-1))


        if self.training:
            loss = masked_lm_loss
            return loss
        else:#eval
            return 0

    def save_pretrained_adapter(self, path):
        self.encoder.save_adaptermodel(path)



class SGroupingAdapterModel_pretrain(nn.Module):

    def __init__(self, PTMwithAdapterModel, fine_grained):

        super().__init__()
        self.hidden = PTMwithAdapterModel.model.config.hidden_size

        self.encoder = PTMwithAdapterModel
        self.tokenizer = PTMwithAdapterModel.tokenizer


        if(fine_grained == True):
            self.linear = nn.Linear(self.hidden, 127)
        else:
            self.linear = nn.Linear(self.hidden, 15)

 
        self.criterion = nn.CrossEntropyLoss()





    def forward(self, input):

        defi_input_ids_batch = input['defi_input_ids_batch']
        defi_mask_batch = input['defi_mask_batch']
        defi_label_batch = input['defi_label_batch']

        defi_emb = self.encoder(input_ids = defi_input_ids_batch, attention_mask = defi_mask_batch)[0] # 0 only one element in tuple output

        defi_emb_cls = defi_emb[:,0,:]

        output = self.linear(defi_emb_cls)
  

        if self.training:
            loss = self.criterion(output, defi_label_batch)

            return loss
        else:#eval
            return 0

    def save_pretrained_adapter(self, path):
        self.encoder.save_adaptermodel(path)



class PTMwithAdapterModel_fusion(nn.Module):

    def __init__(self, device, model, tokenizer, petrained_adaptermodels_pathlist, adapter_pos_list):
        super().__init__()

        self.device = device

        self.tokenizer = tokenizer
        self.model = model

        self.adapters = list()
        self.adapter_pos_list = adapter_pos_list

        for adapterpath in petrained_adaptermodels_pathlist:
            adapter = self.load_adaptermodel(adapterpath)
            adapter.device = self.device
            self.adapters.append(adapter)


        self.hidden = self.adapters[0].adapters[0].project_hidden_size

        self.controller_size = 128



        self.transdim = nn.Linear(768,4096)
        self.dense_size = self.model.config.hidden_size

        self.controller = ControllerModel(self.device, self.hidden, self.controller_size)

        self.dropout = nn.Dropout(p=0.1)


        self.querylist = nn.ModuleList([nn.Linear(self.hidden, self.hidden) for i in range(len(self.adapter_pos_list))])
        self.keylist = nn.ModuleList([nn.Linear(self.hidden, self.hidden) for i in range(len(self.adapter_pos_list))])
        self.valuelist = nn.ModuleList([nn.Linear(self.hidden, self.hidden) for i in range(len(self.adapter_pos_list))])


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask)


        if(self.dense_size<self.hidden):
            outputs_reduced = (self.transdim(outputs[0]),outputs[1], tuple(self.transdim(v) for v in outputs[2]))
        else:
            outputs_reduced = outputs


        pretrained_model_last_hidden_states = outputs_reduced[0]  # original bert/roberta output
        batchsize = pretrained_model_last_hidden_states.size(0) 



        _, controller_hidden_states = self.controller(outputs_reduced)
        

        adapters_outputs = []
        adapters_hidden_states = []

        for adapter in self.adapters:
            adapter_out, adapter_hidden = adapter(outputs_reduced)

            adapters_outputs.append(adapter_out)
            adapters_hidden_states.append(adapter_hidden)
  
        num_adapter_layers = adapters_hidden_states[0].shape[1] #[8,3,128,4096] -> 3

   
        adapter_hidden_states_at_layer = list()
        adapter_hidden_states_at_layer_attentive = list() # weighted sum of adapters


        for i in range(num_adapter_layers):
            hidden_at_layer_i = list()
            for adapterhid in adapters_hidden_states:
                hidden_at_layer_i.append(adapterhid[:,i,:,:]) # layer i in adapterhid

            hidden_at_layer_i_tensor = torch.stack(hidden_at_layer_i, dim = 1)
 
            adapter_hidden_states_at_layer.append(hidden_at_layer_i_tensor)


        attention_scores_prob_batch = []

        for layeri in range(len(adapter_hidden_states_at_layer)):
            layer = adapter_hidden_states_at_layer[layeri] #[8,3,128,4096]

            query = self.querylist[layeri]
            key = self.keylist[layeri]
            value = self.valuelist[layeri]

            controller_hidden_states_at_layer = controller_hidden_states[:,layeri,:,:]  #[8,128,4096,1] #only adapter layer

            query_at_layer = query(controller_hidden_states_at_layer) # # [8,128,4096]

            key_at_layer = key(layer)#[8,3,128,4096]
            key_at_layer = key_at_layer.permute(0,2,1,3)
  

            attention_scores = torch.squeeze(torch.matmul(query_at_layer.unsqueeze(2), key_at_layer.transpose(-2, -1)), dim=2)  #[8,128,3]
            attention_scores = self.dropout(attention_scores)
            attention_probs = F.softmax(attention_scores, dim=-1)

            attention_probs_adapter = torch.mean(attention_probs, dim = 1) # [8,3] -> [3]
            attention_scores_prob_batch.append(attention_probs_adapter)

            value_at_layer = layer
            value_at_layer = value_at_layer.permute(0,2,1,3)

            context_layer = value(torch.squeeze(torch.matmul(attention_probs.unsqueeze(2), value_at_layer), dim=2))

            adapter_hidden_states_at_layer_attentive.append(context_layer)

        attention_scores_prob_batch = torch.stack(attention_scores_prob_batch, dim = 0)
        mean_of_adapters_last = torch.mean(torch.stack(adapters_outputs,1),1)
        task_features = pretrained_model_last_hidden_states + mean_of_adapters_last



        if not self.training:
            return (task_features,), attention_scores_prob_batch

        return (task_features,)

    def save_adaptermodel(self, path):
        self.fac_adapter.save_adaptermodel(path)

    def load_adaptermodel(self, path):
        adapterm = AdapterModel(device = self.device, hiddensize=4096, adaptersize=128).to(self.device)
        adapterm.load_state_dict(torch.load(path, map_location = self.device))
        return adapterm
    