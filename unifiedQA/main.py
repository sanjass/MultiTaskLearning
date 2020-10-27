import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration, T5Model
from transformers.modeling_t5 import load_tf_weights_in_t5
from tqdm import tqdm


data_dir = "../data"
subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(data_dir, "test")) if "_test.csv" in f])

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
base_model = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(base_model)


def get_model(subject, subject_idx, k):
    
    model = T5ForConditionalGeneration(T5Config.from_pretrained(base_model))
    load_tf_weights_in_t5(model, None, "./models")
    print("loaded model")

    if k == 0:
        model.eval()
        model.to(device)
        return model




    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = dict()
            item['input_ids'] = torch.tensor(self.encodings[idx])
            item['attention_mask'] = torch.zeros(len(self.encodings[idx]))
            # print(torch.zeros(len(self.encodings[idx])))
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)


    inputs = []
    labels = []

    dev_df = pd.read_csv(os.path.join(data_dir, "val", subject + "_val.csv"), header=None)
    for i in range(k):
        prompt_end, answers = format_example(dev_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, kshots)
        prompt = train_prompt + prompt_end
        label = dev_df.iloc[i, dev_df.shape[1]-1]
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

    model.to(device)
    model.train()


    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    optim = AdamW(model.parameters(), lr=5e-5)


    for epoch in range(1):
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

    model.eval()
    return model


    

choices = ["A", "B", "C", "D"]
# tokenizer = AutoTokenizer.from_pretrained("t5-base")  
import difflib        

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
    # prompt += "\nAnswer:"
    # if include_answer:
    #     prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt, answers

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}. \\n".format(format_subject(subject))
    # if k == -1:
    #     k = train_df.shape[0]
    # for i in range(k):
    #     prompt += format_example(train_df, i)
    # return prompt
    # return format_example(train_df, 0)
    return ''

kshots = 0
all_scores = []
for subject_i, subject in tqdm(enumerate(subjects)):
    model = get_model(subject, subject_i, kshots)
    dev_df = pd.read_csv(os.path.join(data_dir, "val", subject + "_val.csv"), header=None)
    test_df = pd.read_csv(os.path.join(data_dir, "test", subject + "_test.csv"), header=None)

    correct = 0
    for i in range(test_df.shape[0]):
    # for i in range(1):    
        prompt_end, answers = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, kshots)
        prompt = train_prompt + prompt_end
        label = test_df.iloc[i, test_df.shape[1]-1]

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        res = model.generate(input_ids)
        result = [tokenizer.decode(x) for x in res]
        # print(result)
        # print(result)
        
        # print(closest)
        try:
            closest = difflib.get_close_matches(result[0], answers, n=1)[0]
            if choices.index(label) == answers.index(closest):
                correct += 1
                # print(correct/(i+1))
            # correct += 1
            # print(choices.index(label) == answers.index(result[0]))
        except:
            # print("Notta")
            correct += 0
    print(subject + ": " + str(correct/(i+1)))
    all_scores.append(correct/(i+1))
    model = None
print(all_scores)

sum(all_scores)/len(all_scores)