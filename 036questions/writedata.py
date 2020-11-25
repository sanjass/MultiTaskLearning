import json
import logisticregression as lg1

train_id = 0
test_id = 0

train_data, test_data, test_answers = lg1.return_data(train_id, test_id)
train_id += len(train_data)
test_id += len(test_data)

with open("train.json", 'w') as outfile:
    json.dump(train_data, outfile)
with open('test.json', 'w') as outfile:
    json.dump(test_data, outfile)
with open('answers.json', 'w') as outfile:
    json.dump(test_answers, outfile)
