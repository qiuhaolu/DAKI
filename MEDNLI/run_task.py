import os
import json
import glob
import sys
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (BertConfig, BertModel, BertTokenizer, XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                          XLMConfig, XLMForSequenceClassification, XLMTokenizer, RobertaConfig, RobertaForSequenceClassification,
                          RobertaTokenizer, AlbertConfig, AlbertModel, AlbertTokenizer, AdamW, get_linear_schedule_with_warmup)
from tqdm import tqdm, trange
from utils import convert_examples_to_features, output_modes, processors


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from DAKI.model import *


args = {
    'data_dir': 'data/',
    'model_type': 'albert',
    'model_name': 'albert-xxlarge-v2',
    'task_name': 'binary',
    'output_dir': '/data/qlu1/DAKI/outputs/MEDNLI/',
    'cache_dir': 'cache/',
    'do_train': False,
    'do_eval': True,
    'max_seq_length': 128,
    'output_mode': 'classification',
    'train_batch_size': 16,
    'eval_batch_size': 16,
    'gradient_accumulation_steps': 1,
    'num_train_epochs': 10,
    'weight_decay': 0,
    'learning_rate': 1e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'logging_steps': 50,
    'evaluate_during_training': False,
    'save_steps': 702,
    'eval_all_checkpoints': True,
    'overwrite_output_dir': False,
    'reprocess_input_data': True,
    'notes': 'Using Yelp Reviews dataset'
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Save arguments to a JSON file
with open('args.json', 'w') as f:
    json.dump(args, f)

# Ensure output directory exists
if os.path.exists(args['output_dir']) and os.listdir(args['output_dir']) and args['do_train'] and not args['overwrite_output_dir']:
    raise ValueError(f"Output directory ({args['output_dir']}) already exists and is not empty. Use --overwrite_output_dir to overcome.")

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertModel, AlbertTokenizer)
}

config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]

# Load pretrained model and tokenizer
ptm_model = AlbertModel.from_pretrained('albert-xxlarge-v2', output_hidden_states=True)
tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')

# Load adapter models
adapter_paths = [
    '../pretrained_adaptermodel/albert_umlswithoutdefi_LP_ep2_lr1e5_state_dict.pth',
    '../pretrained_adaptermodel/albert_disease_adaptermodel_ep10_state_dict.pth',
    '../pretrained_adaptermodel/albert_sgrouping_ep1_lr1e5_state_dict.pth'
]
model = PTMwithAdapterModel_fusion(device=device, model=ptm_model, tokenizer=tokenizer, petrained_adaptermodels_pathlist=adapter_paths, adapter_pos_list=[0, 5, 11])
model.to(device)

# Define classification layer
class ClassificationLayer(nn.Module):
    def __init__(self, size):
        super(ClassificationLayer, self).__init__()
        self.f1 = nn.Linear(size, 3)

    def forward(self, x):
        return self.f1(x)

if model.__class__.__name__ in ['PTMwithAdapterModel', 'PTMwithAdapterModel_fusion']:
    classification_layer = ClassificationLayer(model.hidden)
else:
    classification_layer = ClassificationLayer(model.config.hidden_size)

classification_layer.to(device)

# Load dataset
def load_and_cache_examples(task, tokenizer, mode='train'):
    processor = processors[task]()
    cached_features_file = os.path.join(args['data_dir'], f"cached_{mode}_{args['model_name']}_{args['max_seq_length']}_{task}")

    if os.path.exists(cached_features_file) and not args['reprocess_input_data']:
        print(f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        print(f"Creating features from dataset file at {args['data_dir']}")
        examples = {
            'train': processor.get_train_examples,
            'dev': processor.get_dev_examples,
            'test': processor.get_test_examples
        }[mode](args['data_dir'])

        features = convert_examples_to_features(examples, processor.get_labels(), args['max_seq_length'], tokenizer, args['output_mode'],
                                                cls_token_at_end=bool(args['model_type'] in ['xlnet']),
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args['model_type'] in ['xlnet'] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args['model_type'] in ['roberta']),
                                                pad_on_left=bool(args['model_type'] in ['xlnet']),
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args['model_type'] in ['xlnet'] else 0)
        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

# Training loop
def train(train_dataset, model, tower_model, tokenizer):
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args['train_batch_size'])
    t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in ['bias', 'LayerNorm.weight'])], 'weight_decay': args['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in ['bias', 'LayerNorm.weight'])], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps=t_total)
    criterion = nn.CrossEntropyLoss()

    print(f"***** Running training *****\n  Num examples = {len(train_dataset)}\n  Num Epochs = {args['num_train_epochs']}\n  Total train batch size  = {args['train_batch_size']}\n  Total optimization steps = {t_total}")

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()

    for _ in trange(int(args['num_train_epochs']), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None}
            outputs = model(**inputs)
            cls_hidden_state = outputs[0][:, 0]
            logits = tower_model(cls_hidden_state)
            loss = criterion(logits, batch[3])

            if args['gradient_accumulation_steps'] > 1:
                loss /= args['gradient_accumulation_steps']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

            tr_loss += loss.item()
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args['save_steps'] > 0 and global_step % args['save_steps'] == 0:
                    output_dir = os.path.join(args['output_dir'], f'checkpoint-{global_step}')
                    os.makedirs(output_dir, exist_ok=True)
                    torch.save(model, os.path.join(output_dir, 'model.pt'))
                    torch.save(tower_model.state_dict(), os.path.join(output_dir, 'tower_model.pt'))
                    print(f"Saving model checkpoint to {output_dir}")

    return global_step, tr_loss / global_step

# Evaluation loop
def evaluate(model, tower_model, tokenizer, mode, prefix=""):
    eval_dataset = load_and_cache_examples(args['task_name'], tokenizer, mode)
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=args['eval_batch_size'])
    criterion = nn.CrossEntropyLoss()

    print(f"***** Running evaluation {prefix} *****\n  Num examples = {len(eval_dataset)}\n  Batch size = {args['eval_batch_size']}")

    eval_loss = 0.0
    preds = None
    out_label_ids = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None}
            outputs = model(**inputs)
            cls_hidden_state = outputs[0][0][:, 0]
            logits = tower_model(cls_hidden_state)
            tmp_eval_loss = criterion(logits, batch[3])

            eval_loss += tmp_eval_loss.mean().item()
            preds = logits.detach().cpu().numpy() if preds is None else np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = batch[3].detach().cpu().numpy() if out_label_ids is None else np.append(out_label_ids, batch[3].detach().cpu().numpy(), axis=0)

    eval_loss /= len(eval_dataloader)
    preds = np.argmax(preds, axis=1)
    acc = (preds == out_label_ids).mean()

    return {"acc": acc}, eval_loss

def final_train():
    train_dataset = load_and_cache_examples(args['task_name'], tokenizer)
    global_step, tr_loss = train(train_dataset, model, classification_layer, tokenizer)
    print(f"Global step = {global_step}, average loss = {tr_loss}")

def final_test(mode):
    
    if args['eval_all_checkpoints']:
        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args['output_dir'] + '/**/', recursive=True)))

    for checkpoint in checkpoints:
        if not os.path.exists(os.path.join(checkpoint, 'model.pt')):
            continue
        model = torch.load(os.path.join(checkpoint, 'model.pt'))
        classification_layer.load_state_dict(torch.load(os.path.join(checkpoint, 'tower_model.pt')))
        classification_layer.to(device)
        result, _ = evaluate(model, classification_layer, tokenizer, mode, prefix=checkpoint)
        print(f"Eval results for checkpoint {checkpoint}: {result}")

if __name__ == "__main__":
    final_train()
    final_test('test')
