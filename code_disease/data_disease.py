import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, AdamW, BertForMaskedLM
from torch.utils.data import DataLoader
import random

diseaseBERT_data_url = '/disk/luqh/adapter/diseaseKnowledgeInfusionTraining/data/extractedQuestionAnswers_total_removeNoisy_maskedLM.txt'

#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#print(tokenizer.mask_token) #[MASK]
#print(tokenizer.mask_token_id) #103
#print(tokenizer.unk_token_id) #100
#exit()

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results


class dataset_disease(Dataset):
	def __init__(self, file):
		super().__init__()
		self.data = []
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		#self.model = BertForMaskedLM.from_pretrained('/disk/luqh/pretrained_lm/bert-base-uncased',return_dict=True)

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
			#	exit()
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

		return {'token_auxed_passage':token_auxed_passage_ori,
				'token_auxed_passage_ori':token_auxed_passage_ori
				}

	def __len__(self):
		return len(self.data)

 

def batchfn(data):
	token_auxed_passages = [d['token_auxed_passage'] for d in data]
	token_auxed_passages_ori = [d['token_auxed_passage_ori'] for d in data]
	token_auxed_passages = torch.tensor(token_auxed_passages).cuda()
	token_auxed_passages_ori = torch.tensor(token_auxed_passages_ori).cuda()
	return {'token_auxed_passages':token_auxed_passages,
			'token_auxed_passages_ori':token_auxed_passages_ori
	}



'''
dataset = dataset_disease(diseaseBERT_data_url)


dataloader = DataLoader(dataset, batch_size = 16, collate_fn=batchfn)


for batch in dataloader:
	print(batch['token_auxed_passages'])
	exit()

'''