import difflib
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from transformers import AdamW
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration, T5Model
from transformers.modeling_t5 import load_tf_weights_in_t5
     

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0] + "? \\n "
    answers = []
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "({}) {} ".format(choices[j], df.iloc[idx, j+1])
        answers.append(df.iloc[idx, j+1])
    return prompt, answers

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}. \\n".format(format_subject(subject))
    return prompt



class MetaLearner(nn.Module):
    """ Bare Meta-learner class
        Should be added: intialization, hidden states, more control over everything
    """
    def __init__(self, model, k):
        super(MetaLearner, self).__init__()
        self.k = k
        self.weights = nn.Parameter(torch.Tensor(1, 2))

    def forward(self, forward_model, backward_model):
        lr = .001
        f_params = list(forward_model.parameters())
        for i, param in enumerate(backward_model.parameters()):
            cur_grad = (f_params[i].data - param.data) / self.k / lr
            if f_params[i].grad is None:
                f_params[i].grad = V(torch.zeros(cur_grad.size()))
            f_params[i].grad.data.add_(cur_grad)

def train(forward_model, backward_model, optimizer, meta_optimizer, train_data, meta_epochs):
    for meta_epoch in range(meta_epochs): # Meta-training loop (train the optimizer)
        optimizer.zero_grad()
        losses = []
        for batch in train_data:
            forward_model.zero_grad()    
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = forward_model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            losses.append(loss)
            loss.backward()
            optimizer(forward_model, backward_model)
        meta_optimizer.step()
    return forward_model


choices = ["A", "B", "C", "D"]
data_dir = "./"
subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(data_dir, "test")) if "_test.csv" in f])

def get_training_set(subjects, k):
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = dict()
            item['input_ids'] = torch.tensor(self.encodings[idx]).reshape(1, len(self.encodings[idx]))
            item['attention_mask'] = torch.ones(len(self.encodings[idx])).reshape(1, len(self.encodings[idx]))
            item['labels'] = torch.tensor(self.labels[idx]).reshape(1, len(self.labels[idx]))
            return item

        def __len__(self):
            return len(self.labels)
            
    inputs = []
    labels = []
    training_data = []
    for subject in subjects:
        test_df = pd.read_csv(os.path.join(data_dir, "test", subject + "_test.csv"), header=None)
        for i in range(k):
            prompt_end, answers = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            label = test_df.iloc[i, test_df.shape[1]-1]
            if label == "A":
                textlabel = answers[0]
            elif label == "B":
                textlabel = answers[1]
            elif label == "C":
                textlabel = answers[2]
            elif label == "D":
                textlabel = answers[3]    
            inputs.append(tokenizer.encode(prompt))
            labels.append(tokenizer.encode(textlabel))
    train_dataset = Dataset(inputs, labels)
    return train_dataset


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
for k in [0,1,2,5,10,50]:
    base_model = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(base_model)
    model = None
    model = T5ForConditionalGeneration(T5Config.from_pretrained(base_model))
    load_tf_weights_in_t5(model, None, "./")

    if k > 0:
        modelB = T5ForConditionalGeneration(T5Config.from_pretrained(base_model))
        load_tf_weights_in_t5(modelB, None, "./")

        model.to(device)
        model.train()

        modelB.to(device)
        modelB.train()
        opt = MetaLearner(model, k)
        train_dataset = get_training_set(subjects, k)
        good_model = train(model, modelB, opt, AdamW(model.parameters(), lr=5e-5), train_dataset, 10)
        good_model.eval()
        
    else:
        model.to(device)
        good_model = model
        good_model.eval()

    all_scores = []
    for subject_i, subject in enumerate(subjects):
        dev_df = pd.read_csv(os.path.join(data_dir, "val", subject + "_val.csv"), header=None)
        test_df = pd.read_csv(os.path.join(data_dir, "test", subject + "_test.csv"), header=None)

        correct = 0
        for i in range(k, min(test_df.shape[0], 400)):   
            prompt_end, answers = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            label = test_df.iloc[i, test_df.shape[1]-1]

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            res = good_model.generate(input_ids)
            result = [tokenizer.decode(x) for x in res]
            try:
                closest = difflib.get_close_matches(result[0], answers, n=1)[0]
                if choices.index(label) == answers.index(closest):
                    correct += 1
            except:
                correct += 0
        print(subject + ": " + str(correct/(i+1)))
        all_scores.append(correct/(i+1))
        print(sum(all_scores)/len(all_scores))
    print(all_scores)
    print("K: " + str(k))
    print()
    print()
