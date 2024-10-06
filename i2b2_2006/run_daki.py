from __future__ import absolute_import, division, print_function

import glob
import logging
import os
import random
import json
import math
import time

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import random
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange #we should change here later
from tensorboardX import SummaryWriter


from transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer, BertModel, BertForTokenClassification,
                                  XLMConfig, XLMForSequenceClassification, XLMTokenizer, 
                                  XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                                  RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                                  AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer, AlbertModel,AlbertForMaskedLM,
                                  AutoTokenizer, AutoModelForTokenClassification, AutoConfig)

from transformers import AdamW, get_linear_schedule_with_warmup

from utils import (convert_examples_to_features,
                        output_modes, processors)

import torch.nn as nn

from seqeval.metrics import f1_score, precision_score, recall_score


import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from DAKI.model import *




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

args = {
    'data_dir': 'data/',
    'model_type':  'albert',#'albert' 'bert'
    'model_name': 'albert-xxlarge-v2',#'albert-xxlarge-v2''bert-base-uncased'
    'task_name': 'binary',
    'output_dir': '/data/qlu1/DAKI/outputs/i2b2-2006/',
    'cache_dir': 'cache/',
    'do_train': True,
    'do_eval': True,
    'fp16': False,# we have to set it as False
    'fp16_opt_level': 'O1',
    'max_seq_length': 128, # original 128
    'output_mode': 'classification',
    'train_batch_size': 16,
    'eval_batch_size': 16,

    'gradient_accumulation_steps': 2,
    'num_train_epochs': 10,
    'weight_decay': 0,
    'learning_rate': 5e-5, #5e-5
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.1,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,

    'logging_steps': 500,
    'evaluate_during_training': False,
    'save_steps': 1100,
    'eval_all_checkpoints': True,

    'overwrite_output_dir': True,
    'reprocess_input_data': True,
    'notes': 'Using Yelp Reviews dataset'
}

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
#device = torch.device('mps')

with open('args.json', 'w') as f:
    json.dump(args, f)

if os.path.exists(args['output_dir']) and os.listdir(args['output_dir']) and args['do_train'] and not args['overwrite_output_dir']:
    raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args['output_dir']))

MODEL_CLASSES = {
    'bert': (BertConfig, BertForTokenClassification, BertTokenizer),#BertForSequenceClassification
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'albert': (AlbertConfig, AutoModelForTokenClassification, AlbertTokenizer)#AlbertForSequenceClassification   #original AlbertModel
}

config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]


ptm_model = AlbertModel.from_pretrained('albert-xxlarge-v2',output_hidden_states=True)
tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')

adapter1 = '../pretrained_adaptermodel/albert_umlswithoutdefi_LP_ep2_lr1e5_state_dict.pth'
adapter2 = '../pretrained_adaptermodel/albert_disease_adaptermodel_ep10_state_dict.pth'
adapter3 = '../pretrained_adaptermodel/albert_sgrouping_ep1_lr1e5_state_dict.pth'


model = PTMwithAdapterModel_fusion(device = device, model = ptm_model, tokenizer = tokenizer, petrained_adaptermodels_pathlist = [adapter1, adapter2, adapter3], adapter_pos_list = [0,5,11])
#model = PTMwithAdapterModel_fusion(device = device, model = ptm_model, tokenizer = tokenizer, petrained_adaptermodels_pathlist = [adapter2, adapter3], adapter_pos_list = [0,5,11])
#model = PTMwithAdapterModel_fusion(device = device, model = ptm_model, tokenizer = tokenizer, petrained_adaptermodels_pathlist = [adapter1, adapter3], adapter_pos_list = [0,5,11])
#model = PTMwithAdapterModel_fusion(device = device, model = ptm_model, tokenizer = tokenizer, petrained_adaptermodels_pathlist = [adapter1, adapter2], adapter_pos_list = [0,5,11])





class classificationlLayer(nn.Module):
    def __init__(self, size):
        super(classificationlLayer, self).__init__() #the super class of Net is nn.Module, this "super" keywords can search methods in its super class
        self.f1 = nn.Linear(size, 16)

    def forward(self, x):
        x = self.f1(x)
        return x
if(model.__class__.__name__ == 'PTMwithAdapterModel' or model.__class__.__name__ == 'PTMwithAdapterModel_fusion'):
    classificationlLayerMedNLI = classificationlLayer(model.model.config.hidden_size)
else:
    classificationlLayerMedNLI = classificationlLayer(model.config.hidden_size)

classificationlLayerMedNLI.to(device)
model.to(device)

task = args['task_name']

if task in processors.keys() and task in output_modes.keys():
    processor = processors[task]()
    label_list = processor.get_labels()
    num_labels = len(label_list)
else:
    raise KeyError(f'{task} not found in processors or in output_modes. Please check utils.py.')

def load_and_cache_examples(task, tokenizer, mode):
	processor = processors[task]()
	output_mode = args['output_mode']
	#mode = 'dev' if evaluate else 'train'
	cached_features_file = os.path.join(args['data_dir'], f"cached_{mode}_{args['model_name']}_{args['max_seq_length']}_{task}")

	if os.path.exists(cached_features_file) and not args['reprocess_input_data']:
		logger.info("Loading features from cached file %s", cached_features_file)
		features = torch.load(cached_features_file)

	else:
		logger.info("Creating features from dataset file at %s", args['data_dir'])
		label_list = processor.get_labels()
		if mode=='test':
			examples = processor.get_test_examples(args['data_dir'])  
		elif mode=='train':
			examples = processor.get_train_examples(args['data_dir'])
		elif mode=='trainAndDevelopment':
			examples = processor.get_trainDevelopment_examples(args['data_dir'])
		elif mode=='eval':
			examples = processor.get_dev_examples(args['data_dir'])


		if __name__ == "__main__":
			features = convert_examples_to_features(examples, label_list, args['max_seq_length'], tokenizer, output_mode,
                cls_token_at_end=bool(args['model_type'] in ['xlnet']),            # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args['model_type'] in ['xlnet'] else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=bool(args['model_type'] in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=bool(args['model_type'] in ['xlnet']),                 # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args['model_type'] in ['xlnet'] else 0)
		logger.info("Saving features into cached file %s", cached_features_file)
		torch.save(features, cached_features_file)

	all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
	all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

	if output_mode == "classification":
		all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
	elif output_mode == "regression":
		all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
	dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
	return dataset

def train(train_dataset, model, towerModel, tokenizer):
    tb_writer = SummaryWriter()
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args['train_batch_size'])
    
    t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    warmup_steps = math.ceil(t_total * args['warmup_ratio'])
    args['warmup_steps'] = warmup_steps if args['warmup_steps'] == 0 else args['warmup_steps']
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps=t_total)
    criterion = nn.CrossEntropyLoss()

    if args['fp16']:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args['fp16_opt_level'])
        
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args['num_train_epochs'])
    logger.info("  Total train batch size  = %d", args['train_batch_size'])
    logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args['num_train_epochs']), desc="Epoch")
    
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        epoch_loss = 0
        for step, batch in enumerate(epoch_iterator):
            model.train()
            #batch[3] = torch.tensor(batch[3], dtype=torch.float)
            batch = tuple(t.to(device) for t in batch)
            
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      }#'labels':         batch[3]}#
            outputs = model(**inputs)  # PTMwithAdapter outpus is sequence output only, i.e., last layer hidden states, tuple of size 1

            last_hidden_states = outputs[0]#[0]
            logits = towerModel(last_hidden_states)
            labels = batch[3]
            loss = criterion(logits.view(-1, 16), labels.view(-1))
            print("\r%f" % loss, end='')

            epoch_loss += loss

            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])
                
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

            tr_loss += loss.item()
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:
                    # Log metrics
                    if args['evaluate_during_training']:  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args['logging_steps'], global_step)
                    logging_loss = tr_loss

                if args['save_steps'] > 0 and global_step % args['save_steps'] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if(model.__class__.__name__ == 'PTMwithAdapterModel' or model.__class__.__name__ == 'PTMwithAdapterModel_fusion'):
                        torch.save(model, output_dir+'/PTMwithAdapterModel.pt')
                        #exit()
                    else:
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                    tower_PATH = output_dir + 'tower.dict'
                    torch.save(towerModel.state_dict(), tower_PATH)
                    logger.info("Saving model checkpoint to %s", output_dir)
        print('average loss of this epoch: ', epoch_loss*1.0/step)

    return global_step, tr_loss / global_step

from sklearn.metrics import mean_squared_error, matthews_corrcoef, confusion_matrix
from scipy.stats import pearsonr

def get_mismatched(labels, preds):
    mismatched = labels != preds
    examples = processor.get_dev_examples(args['data_dir'])
    wrong = [i for (i, v) in zip(examples, mismatched) if v]
    
    return wrong

def get_eval_report(labels, preds):
    # mcc = matthews_corrcoef(labels, preds)
    # mismatched = labels != preds
    # count_right = 0
    # for item in mismatched:
    #     if item==0:
    #         count_right += 1
    # acc = count_right*1.0/len(labels)
    acc = (preds == labels).mean()
    #tn, fp, fn, tp = 0, 0, 0, 0#confusion_matrix(labels, preds).ravel()
    return {
        "acc": acc,
        # "tp": tp,
        # "tn": tn,
        # "fp": fp,
        # "fn": fn
    }, get_mismatched(labels, preds)

# def compute_metrics(task_name, preds, labels):
#     print(preds[:100])
#     assert len(preds) == len(labels)
#     return get_eval_report(labels, preds)

def align_predictions(predictions: np.ndarray, label_ids: np.ndarray, label_list):

    label_map = {i : label for i, label in enumerate(label_list)}
    #preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = predictions.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[predictions[i][j]])

    return preds_list, out_label_list

def compute_metrics(preds_list, out_label_list):

    return {
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

def evaluate(model, towerModel, tokenizer, mode, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)


    eval_output_dir = args['output_dir']

    results = {}
    EVAL_TASK = args['task_name']

    eval_dataset = load_and_cache_examples(EVAL_TASK, tokenizer, mode)
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)


    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    criterion = nn.CrossEntropyLoss()
    attention_scores_prob_all = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      }#'labels':         batch[3]}#
            outputs, attention_scores_prob_batch = model(**inputs)
            if(attention_scores_prob_all is None):
                attention_scores_prob_all = attention_scores_prob_batch
            else:
                attention_scores_prob_all = torch.cat((attention_scores_prob_all, attention_scores_prob_batch), dim = 1)
            # last_hidden_states = outputs[0]
            # #print(last_hidden_states.size())

            last_hidden_states = outputs[0]#[0]
            logits = towerModel(last_hidden_states)
            labels = batch[3]
            tmp_eval_loss = criterion(logits.view(-1, 16), labels.view(-1))


            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            #out_label_ids = inputs['labels'].detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            #out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    attention_scores_prob_avg = torch.mean(attention_scores_prob_all, dim = 1)
    print(attention_scores_prob_avg)

    eval_loss = eval_loss / nb_eval_steps
    if args['output_mode'] == "classification":
        preds = np.argmax(preds, axis=2)
    elif args['output_mode'] == "regression":
        preds = np.squeeze(preds)
    # preds = np.asarray(preds, dtype=float)
    # out_label_ids = np.asarray(out_label_ids, dtype=float)

    print(preds[:100])
    print(out_label_ids[:100])
    #time.sleep(5)

    preds, out_label_ids = align_predictions(preds, out_label_ids, label_list)
    result = compute_metrics(preds, out_label_ids)

    results.update(result)

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
#with open(output_eval_file, "w") as writer:
    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        #writer.write("%s = %s\n" % (key, str(result[key])))

    return results

def final_train(mode):
    train_dataset = load_and_cache_examples(task, tokenizer, mode)
    global_step, tr_loss = train(train_dataset, model, classificationlLayerMedNLI, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if not os.path.exists(args['output_dir']):
            os.makedirs(args['output_dir'])
    logger.info("Saving model checkpoint to %s", args['output_dir'])

def final_test(mode, withadapters):
    output_eval_file = os.path.join(args['output_dir'], "eval_results_"+mode+".txt")
    writer = open(output_eval_file, "w")

    results = {}
    result_list = []
    checkpoints = [args['output_dir']]
    if args['eval_all_checkpoints'] and withadapters == 'PTMwithAdapterModel':
        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args['output_dir'] + '/**/', recursive=True)))
        #print(len(checkpoints[1:]))
        checkpoints = checkpoints[1:]
        #exit()
    elif args['eval_all_checkpoints'] and withadapters:
        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args['output_dir'] + '/**/' + WEIGHTS_NAME, recursive=True)))
        logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging


    logger.info("Evaluate the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
        global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""

        if(withadapters == 'PTMwithAdapterModel'):
            model = torch.load(checkpoint+'/PTMwithAdapterModel.pt')
        else:
            model = model_class.from_pretrained(checkpoint)
        tower_PATH = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step)) + 'tower.dict'
        classificationlLayerMedNLI.load_state_dict(torch.load(tower_PATH))
        model.to(device)
        classificationlLayerMedNLI.to(device)

        result = evaluate(model, classificationlLayerMedNLI, tokenizer, mode, prefix=global_step)
        result_list.append(result)
        result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
        results.update(result)

        logger.info("***** Eval results {} *****".format(global_step))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
            writer.flush()
    print(result_list)

if __name__ == "__main__":
    final_train('train')

    final_test('test', 'PTMwithAdapterModel')