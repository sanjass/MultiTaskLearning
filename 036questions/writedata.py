import json
import random
import logisticregression as lg1
import gradientdescent as gd1
import gradientdescent2 as gd2
import filtering as f1
import filtering2 as f2

train_id = 0
test_id = 0
train_data = []
test_data = []
test_answers = []
modules = [lg1, gd1, gd2, f1, f2]

def get_and_update(mod, train_id, test_id, train_data, test_data, test_answers):
    train, test, answers = mod.return_data(train_id, test_id)
    return train_id + len(train_data), test_id + len(test_data), train_data + train, test_data+test, test_answers + answers

for mod in modules:
    train_id, test_id, train_data, test_data, test_answers = get_and_update(mod, train_id, test_id, train_data, test_data, test_answers)

random.shuffle(train_data)
with open("train.json", 'w') as outfile:
    json.dump(train_data, outfile)
with open('test.json', 'w') as outfile:
    json.dump(test_data, outfile)
with open('answers.json', 'w') as outfile:
    json.dump(test_answers, outfile)
