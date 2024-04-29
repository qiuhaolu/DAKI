import torch.nn as nn
import torch
import pickle
from math import floor
import os
from transformers import BertTokenizer, BertModel, AdamW, AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer, AlbertModel,AlbertForMaskedLM, AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from transformers.modeling_albert import AlbertMLMHead

from torch.nn import functional as F
from torch.optim import Adam

class Adapter(nn.Module):
    #def __init__(self, args,adapter_config):
    def __init__(self, hiddensize, adaptersize):
        super(Adapter, self).__init__()
        #self.adapter_config = adapter_config
        #self.args = args
        self.adapter_size = adaptersize #default 128
        self.project_hidden_size = hiddensize #bert-base 768 albert-xxlarge 4096
        self.down_project = nn.Linear(self.project_hidden_size,self.adapter_size)
        #self.encoder = BertEncoder(self.adapter_config)
        self.up_project = nn.Linear(self.adapter_size, self.project_hidden_size)

    def forward(self, hidden_states):
        down_projected = F.leaky_relu(self.down_project(hidden_states))
        up_projected = self.up_project(down_projected)

        #input_shape = down_projected.size()[:-1]
        #attention_mask = torch.ones(input_shape, device=self.args.device)
        #encoder_attention_mask = torch.ones(input_shape, device=self.args.device)
        #if attention_mask.dim() == 3:
        #    extended_attention_mask = attention_mask[:, None, :, :]
        #if attention_mask.dim() == 2:
        #    extended_attention_mask = attention_mask[:, None, None, :]
        #extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        #extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        #if encoder_attention_mask.dim() == 3:
        #    encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        #if encoder_attention_mask.dim() == 2:
        #    encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        #head_mask = [None] * self.adapter_config.num_hidden_layers
        #encoder_outputs = self.encoder(down_projected,
        #                               attention_mask=extended_attention_mask,
        #                               head_mask=head_mask)

        #up_projected = self.up_project(encoder_outputs[0])
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

        #self.reducedim = nn.Linear(4096,768)

    def forward(self, pretrained_model_outputs):
        outputs = pretrained_model_outputs

        #print(outputs[2].shape)
        #print(outputs.hidden_states)

        sequence_output = outputs[0]
        # pooler_output = outputs[1]
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
            #adapter_hidden_states_count += 1
            #if self.adapter_skip_layers >= 1:
            #    if adapter_hidden_states_count % self.adapter_skip_layers == 0:
            #        hidden_states_last = hidden_states_last + adapter_hidden_states[int(adapter_hidden_states_count/self.adapter_skip_layers)]
        #print(outputs[2:])
        #outputs = (hidden_states_last,) + outputs[2:] # latter not needed

        adapter_hidden_states_tensor = torch.stack(adapter_hidden_states, dim = 1)



        outputs = hidden_states_last
        #print(hidden_states_last.shape)
        #print(len(hidden_states))
        #print(hidden_states[0].shape)
        #print(outputs.shape)
        #exit()
        return outputs, adapter_hidden_states_tensor  # (loss), logits, (hidden_states), (attentions)

    def save_adaptermodel(self, path):
        #assert os.path.isdir(path)
        torch.save(self, path)
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

        #print(outputs[2].shape)
        #print(outputs.hidden_states)

        sequence_output = outputs[0]    
        # pooler_output = outputs[1]
        hidden_states = outputs[2]


        num = len(hidden_states)
        hidden_states_last = torch.zeros(sequence_output.size()).to(self.device)  # task context vec: initialize with 0s #[8,128,4096]
        #hidden_states_last = taskvecbatch
        #print(hidden_states_last.shape)
        #exit()
        #print(self.device)

        adapter_hidden_states = []
        adapter_hidden_states_count = 0
        for i, adapter_module in enumerate(self.adapters):
            fusion_state = hidden_states[self.adapter_list[i]] + hidden_states_last
            hidden_states_last = adapter_module(fusion_state)
            adapter_hidden_states.append(hidden_states_last)
            #adapter_hidden_states_count += 1
            #if self.adapter_skip_layers >= 1:
            #    if adapter_hidden_states_count % self.adapter_skip_layers == 0:
            #        hidden_states_last = hidden_states_last + adapter_hidden_states[int(adapter_hidden_states_count/self.adapter_skip_layers)]
        #print(outputs[2:])
        #outputs = (hidden_states_last,) + outputs[2:] # latter not needed

        adapter_hidden_states_tensor = torch.stack(adapter_hidden_states, dim = 1)



        outputs = hidden_states_last
        #print(hidden_states_last.shape)
        #print(len(hidden_states))
        #print(hidden_states[0].shape)
        #print(outputs.shape)
        #exit()
        return outputs, adapter_hidden_states_tensor  # (loss), logits, (hidden_states), (attentions)







class PTMwithAdapterModel(nn.Module):

    #def __init__(self, vocab_size, emb_file, relation_size, hidden=200, dropout=0.1, margin=0.1, p_norm=2):
    def __init__(self, device, model, tokenizer, adaptersize = 128, petrained_adaptermodel_path = None, freeze_encoder = True, freeze_adapter = False, fusion_mode = 'add'):
        super().__init__()
        #self.hidden = 768
        #self.p_norm = 2
        #self.margin = 1.0
        self.device = device
        self.fusion_mode = fusion_mode
        self.freeze_encoder = freeze_encoder
        self.freeze_adapter = freeze_adapter
        #bert, albert, clinicalbert, roberta ...
        #self.tokenizer = BertTokenizer.from_pretrained('/disk/luqh/pretrained_lm/bert-base-uncased')
        #self.model = BertModel.from_pretrained('/disk/luqh/pretrained_lm/bert-base-uncased',output_hidden_states=True)
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
            #self.fac_adapter.to(self.device)
        #print(self.fac_adapter.adapter_list)

        if self.freeze_adapter:
            for p in self.fac_adapter.parameters():
                p.requires_grad = False

        ####add/concat mode code######






    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
    #def forward(self, **inputs):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask)

        #outputs = self.model(**inputs)


        pretrained_model_last_hidden_states = outputs[0]  # original bert/roberta output

        #print(outputs)
        #print(outputs[0].shape)
        #exit()

        #fac_adapter_outputs, _ = self.fac_adapter(outputs) # latter not needed
        fac_adapter_outputs, _ = self.fac_adapter(outputs)

        task_features = pretrained_model_last_hidden_states

        if self.fusion_mode == 'add':
            task_features = task_features + fac_adapter_outputs

        #print(task_features.shape)
        #exit()
        #elif self.args.fusion_mode == 'concat':
        #    combine_features = pretrained_model_last_hidden_states
        #    if self.args.meta_fac_adaptermodel:
        #        fac_features = self.task_dense_fac(torch.cat([combine_features, fac_adapter_outputs], dim=2))
        #        task_features = fac_features
        #    if self.args.meta_lin_adaptermodel:
        #        lin_features = self.task_dense_lin(torch.cat([combine_features, lin_adapter_outputs], dim=2))
        #        task_features = lin_features
        #    if (self.fac_adapter is not None) and (self.lin_adapter is not None):
        #        task_features = self.task_dense(torch.cat([fac_features, lin_features], dim=2))
        return (task_features,)

    def save_adaptermodel(self, path):
        self.fac_adapter.save_adaptermodel(path)

    def load_adaptermodel(self, path):
        return torch.load(path, map_location = self.device)



class UMLSAdapterModel_pretrain(nn.Module):

    #def __init__(self, vocab_size, emb_file, relation_size, hidden=200, dropout=0.1, margin=0.1, p_norm=2):
    def __init__(self, relation_size, PTMwithAdapterModel):

        super().__init__()
        self.hidden = PTMwithAdapterModel.model.config.hidden_size
        self.p_norm = 2
        self.margin = 0.5

        #bert, albert, clinicalbert, roberta ...
        #self.tokenizer = BertTokenizer.from_pretrained('/disk/luqh/pretrained_lm/bert-base-uncased')
        #self.encoder = BertModel.from_pretrained('/disk/luqh/pretrained_lm/bert-base-uncased')
        
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

        #print(p_score)
        #exit()






        if self.training:
            loss = loss = self.criterion(p_score, n_score, torch.tensor([-1.0]*batch_size).cuda())

            return loss
        else:#eval
            return 0

    def save_pretrained_adapter(self, path):
        self.encoder.save_adaptermodel(path)

    #def save_model_with_adapter(self, url):
    #    self.encoder.save_pretrained(url)
    #    self.encoder.save_adapter(url, self.adaptername)









class UMLSAdapterModel_LP_pretrain(nn.Module):

    #def __init__(self, vocab_size, emb_file, relation_size, hidden=200, dropout=0.1, margin=0.1, p_norm=2):
    def __init__(self, PTMwithAdapterModel):

        super().__init__()
        self.hidden = PTMwithAdapterModel.model.config.hidden_size
        #self.p_norm = 2
        #self.margin = 0.5

        #bert, albert, clinicalbert, roberta ...
        #self.tokenizer = BertTokenizer.from_pretrained('/disk/luqh/pretrained_lm/bert-base-uncased')
        #self.encoder = BertModel.from_pretrained('/disk/luqh/pretrained_lm/bert-base-uncased')
        
        self.encoder = PTMwithAdapterModel
        self.tokenizer = PTMwithAdapterModel.tokenizer

        self.linear = nn.Linear(self.hidden, 2)

        #self.relation_embedding = nn.Embedding(num_embeddings=relation_size, embedding_dim=self.hidden)
        # initialization
        #nn.init.xavier_normal_(self.relation_embedding.weight)

        #self.criterion = nn.MarginRankingLoss(self.margin)
        self.criterion = nn.CrossEntropyLoss()





    def forward(self, input):
        '''
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

        #print(p_score)
        #exit()
        '''
        triple_pos_ids_s = input['triple_pos_ids_s']
        triple_pos_mask_s = input['triple_pos_mask_s']
        triple_neg_ids_s = input['triple_neg_ids_s']
        triple_neg_mask_s = input['triple_neg_mask_s']

        #print(len(triple_pos_ids_s))
        #exit()

        triple_pos_emb = self.encoder(input_ids = triple_pos_ids_s, attention_mask = triple_pos_mask_s)[0] # 0 only one element in tuple output
        triple_neg_emb = self.encoder(input_ids = triple_neg_ids_s, attention_mask = triple_neg_mask_s)[0]

        #print(triple_pos_emb.shape)
        #exit()
        triple_pos_neg_cls = torch.cat([triple_pos_emb[:,0,:],triple_neg_emb[:,0,:]])
        #print(triple_pos_neg_cls.shape) # [batch*2,4096]
        label = torch.tensor([1] * len(triple_pos_ids_s) + [0] * len(triple_neg_ids_s)).cuda()
        #print(label)
        #print(label.shape)


        #print(triple_pos_cls.shape)
        output = self.linear(triple_pos_neg_cls)
        #print(output.shape)

        #exit()







        if self.training:
            #loss = loss = self.criterion(p_score, n_score, torch.tensor([-1.0]*batch_size).cuda())
            loss = self.criterion(output, label)
            #print(loss)
            #print(loss.shape)
            #exit()

            return loss
        else:#eval
            return 0

    def save_pretrained_adapter(self, path):
        self.encoder.save_adaptermodel(path)

    #def save_model_with_adapter(self, url):
    #    self.encoder.save_pretrained(url)
    #    self.encoder.save_adapter(url, self.adaptername)





















class diseaseAdapterModel_pretrain(nn.Module):

    #def __init__(self, vocab_size, emb_file, relation_size, hidden=200, dropout=0.1, margin=0.1, p_norm=2):
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

        #print(masked_lm_loss)
        #exit()




        #prediction_scores = self.predictions(sequence_outputs)



        #print(token_auxed_passages.shape)
        #outputs = self.model(input_ids = token_auxed_passages, labels = token_auxed_passages_ori)
        #loss = outputs.loss
        #logits = outputs.logits





        if self.training:
            #loss = loss = self.criterion(p_score, n_score, torch.tensor([-1.0]*batch_size).cuda())
            loss = masked_lm_loss
            return loss
        else:#eval
            return 0

    def save_pretrained_adapter(self, path):
        self.encoder.save_adaptermodel(path)

    #def save_model_with_adapter(self, url):
    #    self.encoder.save_pretrained(url)
    #    self.encoder.save_adapter(url, self.adaptername)



class SGroupingAdapterModel_pretrain(nn.Module):

    def __init__(self, PTMwithAdapterModel, fine_grained):

        super().__init__()
        self.hidden = PTMwithAdapterModel.model.config.hidden_size
        #self.p_norm = 2
        #self.margin = 0.5

        #bert, albert, clinicalbert, roberta ...
        #self.tokenizer = BertTokenizer.from_pretrained('/disk/luqh/pretrained_lm/bert-base-uncased')
        #self.encoder = BertModel.from_pretrained('/disk/luqh/pretrained_lm/bert-base-uncased')
        
        self.encoder = PTMwithAdapterModel
        self.tokenizer = PTMwithAdapterModel.tokenizer


        if(fine_grained == True):
            self.linear = nn.Linear(self.hidden, 127)
        else:
            self.linear = nn.Linear(self.hidden, 15)

        #self.relation_embedding = nn.Embedding(num_embeddings=relation_size, embedding_dim=self.hidden)
        # initialization
        #nn.init.xavier_normal_(self.relation_embedding.weight)

        #self.criterion = nn.MarginRankingLoss(self.margin)
        self.criterion = nn.CrossEntropyLoss()





    def forward(self, input):

        defi_input_ids_batch = input['defi_input_ids_batch']
        defi_mask_batch = input['defi_mask_batch']
        defi_label_batch = input['defi_label_batch']

        #print(len(triple_pos_ids_s))
        #exit()

        defi_emb = self.encoder(input_ids = defi_input_ids_batch, attention_mask = defi_mask_batch)[0] # 0 only one element in tuple output


        #print(defi_emb.shape)
        #exit()

        defi_emb_cls = defi_emb[:,0,:]
        #print(defi_emb_cls.shape)


        #triple_pos_neg_cls = torch.cat([triple_pos_emb[:,0,:],triple_neg_emb[:,0,:]])
        #print(triple_pos_neg_cls.shape) # [batch*2,4096]
        #label = torch.tensor([1] * len(triple_pos_ids_s) + [0] * len(triple_neg_ids_s)).cuda()
        #print(label)
        #print(label.shape)


        #print(triple_pos_cls.shape)
        output = self.linear(defi_emb_cls)
        #print(output.shape)

        #exit()







        if self.training:
            #loss = loss = self.criterion(p_score, n_score, torch.tensor([-1.0]*batch_size).cuda())
            loss = self.criterion(output, defi_label_batch)
            #print(loss)
            #print(loss.shape)
            #exit()

            return loss
        else:#eval
            return 0

    def save_pretrained_adapter(self, path):
        self.encoder.save_adaptermodel(path)


'''
a11 a12 a13

a51 a52 a53

t

a1 = f(a11, a12, a13, t1)
a5 = f(a51, a52, a53, t5)

….

ti = g(t_{i-1}, a_{i-1})

t1 = g(t0)

t5 = g(t1, a1)

t11 = g(t5,a5)




a11 a12 a13

b = f(a11_1, a12_1, a13_1, …., a11_n, a12_n, a13_n)

a11 a12 a13

a1 = g(b, a11, a12, a13)


'''

# fusion doesn't need pretraining support, so no adaptersize, no fusion_mode
class PTMwithAdapterModel_fusion(nn.Module):

    #def __init__(self, device, model, tokenizer, adaptersize = 128, petrained_adaptermodel_path = None, freeze_encoder = True, freeze_adapter = False, fusion_mode = 'add'):
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

        #self.hidden = self.model.config.hidden_size
        self.hidden = self.adapters[0].adapters[0].project_hidden_size
        #print(self.hidden)
        #exit()
        self.controller_size = 128

        #self.task_context_vector = nn.Parameter(torch.normal(0, 0.05, size = (self.controller_size, self.hidden)))
        #print(self.task_context_vector)
        #exit()

        

        self.transdim = nn.Linear(768,4096)
        self.dense_size = self.model.config.hidden_size

        self.controller = ControllerModel(self.device, self.hidden, self.controller_size)

        self.dropout = nn.Dropout(p=0.1)

        #self.query = nn.Linear(self.hidden, self.dense_size)
        #self.key = nn.Linear(self.dense_size, self.dense_size)
        #self.value = nn.Linear(self.hidden, self.hidden)

        self.querylist = nn.ModuleList([nn.Linear(self.hidden, self.hidden) for i in range(len(self.adapter_pos_list))])
        self.keylist = nn.ModuleList([nn.Linear(self.hidden, self.hidden) for i in range(len(self.adapter_pos_list))])
        self.valuelist = nn.ModuleList([nn.Linear(self.hidden, self.hidden) for i in range(len(self.adapter_pos_list))])


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
    #def forward(self, **inputs):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask)


        #pretrained_model_last_hidden_states = outputs[0]  # original bert/roberta output
        #batchsize = pretrained_model_last_hidden_states.size(0) 
        #task_context_vector_batch = torch.stack(batchsize*[self.task_context_vector], dim = 0)

        #print(task_context_vector_batch.shape) #[8,128,4096]
        #print(task_context_vector_batch)
        #exit()
        #print(batchsize)
        #exit()
        if(self.dense_size<self.hidden):
            outputs_reduced = (self.transdim(outputs[0]),outputs[1], tuple(self.transdim(v) for v in outputs[2]))
        else:
            outputs_reduced = outputs


        pretrained_model_last_hidden_states = outputs_reduced[0]  # original bert/roberta output
        batchsize = pretrained_model_last_hidden_states.size(0) 



        _, controller_hidden_states = self.controller(outputs_reduced)
        #controller_hidden_states = torch.stack(outputs[2], dim = 1)

        #print(len(outputs[2]))
        #print(outputs[2][0].shape)
        #exit()

        ###test Q K V at layer
        '''
        controller_hidden_states_at_layer = controller_hidden_states[:,0,:,:] # # [8,128,4096]
        #print(controller_hidden_states_at_layer.shape)
        query_at_layer = self.query(controller_hidden_states_at_layer)
        #print(query_at_layer.shape)

        key_at_layer = self.key(controller_hidden_states_at_layer)
        value_at_layer = self.value(controller_hidden_states_at_layer)
        #print(query_at_layer.unsqueeze(2).shape)
        #print(key_at_layer.transpose(-2, -1).shape)

        attention_scores = torch.matmul(query_at_layer, key_at_layer.transpose(-2, -1))
        #print(attention_scores.shape)

        attention_probs = F.softmax(attention_scores, dim=-1)
        #print(attention_probs)
        #print(attention_probs.shape)

        context_layer = torch.matmul(attention_probs, value_at_layer)
        print(context_layer.shape)
        exit()
        '''
        
        #print(controller_hidden_states.shape) #[8,13,128,4096]
        #exit()
        #print(outputs)
        #print(outputs[2].shape)
        #exit()

        #print(outputs[0].shape)
        #exit()
        #outputs[0] = self.transdim(outputs[0])
        #outputs[2] = self.transdim(outputs[2])

        

        #exit()


        adapters_outputs = []
        adapters_hidden_states = []

        for adapter in self.adapters:
            adapter_out, adapter_hidden = adapter(outputs_reduced)

            adapters_outputs.append(adapter_out)
            adapters_hidden_states.append(adapter_hidden)
            #print(adapter_out)
            #print('aaa')
            #print(adapter_hidden)
            #exit()
            #print(adapter_out.shape) #[8,128,4096]
            #print(len(adapter_hidden)) #layer = 3, it refers to the number of layers of each adapter, not the types of adapters, which is also 3
            #print(adapter_hidden.shape) #[8,3,128,4096]

        #a11*c

        num_adapter_layers = adapters_hidden_states[0].shape[1] #[8,3,128,4096] -> 3

        #print(num_adapter_layers) # 3
        #exit()

        adapter_hidden_states_at_layer = list()
        adapter_hidden_states_at_layer_attentive = list() # weighted sum of adapters


        for i in range(num_adapter_layers):
            hidden_at_layer_i = list()
            for adapterhid in adapters_hidden_states:
                hidden_at_layer_i.append(adapterhid[:,i,:,:]) # layer i in adapterhid
                #print(adapterhid.shape)
                #print(adapterhid[i].shape)
                #print(adapterhid[:,i,:,:].shape)
            #print(len(hidden_at_layer_i))
            hidden_at_layer_i_tensor = torch.stack(hidden_at_layer_i, dim = 1)
            #print(hidden_at_layer_i_tensor.shape)
            #exit()
            adapter_hidden_states_at_layer.append(hidden_at_layer_i_tensor)

        #print(len(adapter_hidden_states_at_layer)) #3
        #exit()



        attention_scores_prob_batch = []

        for layeri in range(len(adapter_hidden_states_at_layer)):
            layer = adapter_hidden_states_at_layer[layeri] #[8,3,128,4096]
            #layer_23reverse = layer.permute(0,2,1,3) #[8,128,3,4096]
            #print(layer_23reverse.shape)
            #exit()

            ### new fusion
            
            query = self.querylist[layeri]
            key = self.keylist[layeri]
            value = self.valuelist[layeri]

            #controller_hidden_states_at_layer = controller_hidden_states[:,self.adapter_pos_list[layeri],:,:] # # [8,128,4096]  # every layer
            controller_hidden_states_at_layer = controller_hidden_states[:,layeri,:,:]  #[8,128,4096,1] #only adapter layer


            query_at_layer = query(controller_hidden_states_at_layer) # # [8,128,4096]
            #query_at_layer = controller_hidden_states_at_layer
            #print(query_at_layer.shape)

            key_at_layer = key(layer)#[8,3,128,4096]
            #key_at_layer = layer
            key_at_layer = key_at_layer.permute(0,2,1,3)
            #print(key_at_layer.shape)
            #exit()

            #attention_scores = torch.matmul(query_at_layer.unsqueeze(1), key_at_layer.transpose(-2, -1))  #[8,3,128,128]
            #print(attention_scores.shape)
            attention_scores = torch.squeeze(torch.matmul(query_at_layer.unsqueeze(2), key_at_layer.transpose(-2, -1)), dim=2)  #[8,128,3]

            #attention_adapter = torch.mean(attention_scores, dim = 1) # [8,3]
            #print(attention_adapter)
            #print(attention_adapter.shape)

            attention_scores = self.dropout(attention_scores)

            #print(attention_scores.shape)
            #exit()

            
            attention_probs = F.softmax(attention_scores, dim=-1)
            #print(attention_probs)
            #print(attention_probs.shape)
            attention_probs_adapter = torch.mean(attention_probs, dim = 1) # [8,3] -> [3]


            #print(attention_probs_adapter)
            #print(attention_probs_adapter.shape)
            attention_scores_prob_batch.append(attention_probs_adapter)


            #exit()
            #value_at_layer = value(layer)#[8,3,128,4096]
            value_at_layer = layer
            value_at_layer = value_at_layer.permute(0,2,1,3)

            #context_layer = torch.matmul(attention_probs, value_at_layer) #[8,3,128,4096]
            #context_layer = torch.sum(context_layer, dim = 1)
            #print(context_layer.shape)

            context_layer = value(torch.squeeze(torch.matmul(attention_probs.unsqueeze(2), value_at_layer), dim=2))
            #print(context_layer.shape)
            #exit()


            adapter_hidden_states_at_layer_attentive.append(context_layer)

            
            #exit()
            



            ### ori before 4.17
            '''
            # mean or [CLS], need test
            layer_sequence_mean = torch.mean(layer, dim = 2) #[8,3,4096]
            #layer_sequence_cls = layer[:,:,0,:] #[8,3,4096]


            controller_ = controller_hidden_states[:,self.adapter_pos_list[layeri],:,:].unsqueeze(3)   #[8,128,4096,1]  #every layer
            #controller_ = controller_hidden_states[:,layeri,:,:].unsqueeze(3)   #[8,128,4096,1] #only adapter layer

            controller_mean = torch.mean(controller_, dim = 1) #[8,4096,1]
            #controller_cls = controller_[:,0,:] #[8,4096,1]


            attention_score_mean = torch.matmul(layer_sequence_mean, controller_mean) #[8,3,1]
            attention_score_mean = F.softmax(attention_score_mean, dim = 1)

            #attention_score_cls = torch.matmul(layer_sequence_cls, controller_cls) #[8,3,1]
            #attention_score_cls = F.softmax(attention_score_cls, dim = 1)


            layer_sequence_cls_attentive = torch.mul(layer.view(batchsize,3,-1), attention_score_mean)
            #layer_sequence_cls_attentive = torch.mul(layer.view(batchsize,3,-1), attention_score_cls)


            layer_sequence_cls_attentive = layer_sequence_cls_attentive.view(batchsize,3,-1,self.hidden)
            layer_sequence_cls_attentive = torch.sum(layer_sequence_cls_attentive, dim = 1)

            adapter_hidden_states_at_layer_attentive.append(layer_sequence_cls_attentive)



            '''


        attention_scores_prob_batch = torch.stack(attention_scores_prob_batch, dim = 0)
        #print(attention_scores_prob_batch.shape)
        #print(attention_scores_prob_batch)
        #exit()










        

        #if self.fusion_mode == 'add':
        #1   #sucks
        #directly add the last layer of attentive adapters, without considering the first couple of layers

        #task_features = pretrained_model_last_hidden_states + adapter_hidden_states_at_layer_attentive[len(adapter_hidden_states_at_layer_attentive)-1]
        #print(task_features.shape)
        #exit()

        #2 add an adapter of adapters
        #task_features = torch.cat(adapter_hidden_states_at_layer_attentive, dim = 2)
        #task_features = self.concatadapters(task_features)
        #task_features = task_features + pretrained_model_last_hidden_states
        #print(task_features.shape)
        #exit()


        #3
        #print(adapters_outputs[0].shape)
        #mean_of_adapters_last = torch.mean(torch.stack(adapters_outputs,1),1)
        #task_features = pretrained_model_last_hidden_states + mean_of_adapters_last + adapter_hidden_states_at_layer_attentive[len(adapter_hidden_states_at_layer_attentive)-1]
        #print(mean_of_adapters_last.shape)
        #exit()

        #3 with task convext vec
        #print(adapters_outputs[0].shape)
        mean_of_adapters_last = torch.mean(torch.stack(adapters_outputs,1),1)
        #print(pretrained_model_last_hidden_states.shape)
        #print(mean_of_adapters_last.shape)
        #print(adapter_hidden_states_at_layer_attentive[len(adapter_hidden_states_at_layer_attentive)-1].shape)
        #task_features = pretrained_model_last_hidden_states + mean_of_adapters_last + adapter_hidden_states_at_layer_attentive[len(adapter_hidden_states_at_layer_attentive)-1]
        task_features = pretrained_model_last_hidden_states + mean_of_adapters_last
        #print(mean_of_adapters_last.shape)
        #exit()


        #4
        #print(adapters_outputs[0].shape)
        #mean_of_adapters_last = torch.mean(torch.stack(adapters_outputs,1),1)  # SUM doesnt work
        #task_features = pretrained_model_last_hidden_states
        #print(mean_of_adapters_last.shape)
        #exit()


        #5
        #print(adapters_outputs[0].shape)
        #mean_of_adapters_last = torch.mean(torch.stack(adapters_outputs,1),1)
        #task_features = pretrained_model_last_hidden_states + adapter_hidden_states_at_layer_attentive[len(adapter_hidden_states_at_layer_attentive)-1]
        #print(mean_of_adapters_last.shape)
        #exit()

        
        #6
        #print(adapters_outputs[0].shape)
        #task_features = torch.cat([pretrained_model_last_hidden_states,adapter_hidden_states_at_layer_attentive[len(adapter_hidden_states_at_layer_attentive)-1]], dim = 2)
        #task_features = self.concatadapters6(task_features)
        #print(task_features.shape)
        #exit()

        #7 sucks
        #print(adapters_outputs[0].shape)
        #mean_of_adapters_last = torch.mean(torch.stack(adapters_outputs,1),1)
        #task_features = torch.cat([pretrained_model_last_hidden_states,mean_of_adapters_last,adapter_hidden_states_at_layer_attentive[len(adapter_hidden_states_at_layer_attentive)-1]], dim = 2)
        #task_features = self.concatadapters7(task_features)
        #print(task_features.shape)
        #exit()

        #### put controller only at the adapter layers works better than at every layer
        #### not sure

        #### using task vec in ControllerModel hurts

        #### use raw PTM hidden states as query, instead of incorporating the ControllerModel, let alone the task vec
        #### sucks

        ####


        #control over adapter layer,  dropout 0.5  works for trecqa2017, dropout0.3 works for mediqa2019  0.1 for NCBI/MEDNLI

        #t0 - bc5cdr - w/o  - dropout0.5 to test attention
        #t1 - mednli - w/o controller+sg - dropout0.3
        #t2 - mediqa - w/o controller+sg - dropout0.3




        #exit()

        if not self.training:
            return (task_features,), attention_scores_prob_batch

        return (task_features,)

    def save_adaptermodel(self, path):
        self.fac_adapter.save_adaptermodel(path)

    def load_adaptermodel(self, path):
        return torch.load(path, map_location = self.device)