import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, AdamW, BertForMaskedLM
from torch.utils.data import DataLoader
from data_disease import *
from transformers import AdapterConfig
from transformers import AdapterType


diseaseBERT_data_url = '/disk/luqh/adapter/diseaseKnowledgeInfusionTraining/data/extractedQuestionAnswers_total_removeNoisy_maskedLM.txt'
save_adapter_url = '/disk/luqh/adapter/pretrained_adapters/disease/'

torch.manual_seed(1)
torch.cuda.set_device(1)



class diseaseAdapter(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		self.model = BertForMaskedLM.from_pretrained('/disk/luqh/pretrained_lm/bert-base-uncased',return_dict=True)
		self.adaptername = 'disease'
		task_config = AdapterConfig.load("pfeiffer", reduction_factor=12)
		self.model.add_adapter(self.adaptername, AdapterType.text_task, config=task_config)

		self.model.train_adapter([self.adaptername])
		self.model.set_active_adapters(self.adaptername)

	def forward(self, x):
		token_auxed_passages = x['token_auxed_passages']
		token_auxed_passages_ori = x['token_auxed_passages_ori']
		#print(token_auxed_passages.shape)
		outputs = self.model(input_ids = token_auxed_passages, labels = token_auxed_passages_ori)
		loss = outputs.loss
		logits = outputs.logits
		#print(loss)

		return loss

	def save_model_with_adapter(self, url):
		self.model.save_pretrained(url)
		# save adapter
		#self.model.save_adapter(url, self.adaptername) # no need 



dataset = dataset_disease(diseaseBERT_data_url)
trainloader = DataLoader(dataset, batch_size = 16, collate_fn=batchfn, pin_memory = False)
model = diseaseAdapter().cuda()
optimizer = AdamW(model.parameters(), lr = 2e-5)
#train
model.train()
print('start training...')

for batch in trainloader:
	optimizer.zero_grad()
	loss = model(batch)
	loss.backward()
	optimizer.step()

	#print(loss)
	
	#exit()

model.save_model_with_adapter(save_adapter_url)



