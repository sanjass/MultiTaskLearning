import csv
import os
import json

def check_numeric(question):
    for i in range(1, 5):
        if not question[i].isnumeric():
            return False
    return True

def gen_quant_cells(question):
    lst = []
    for i in range(len(question[0].split(" "))):
        lst += [i+1]
    return lst

def gen_answer(question):
    answer = question[5]
    if answer == "A":
        return question[1]
    if answer == "B":
        return question[2]
    if answer == "C":
        return question[3]
    if answer == "D":
        return question[4]

dir_path = os.path.dirname(os.path.realpath(__file__))
directory = dir_path + "/relevant_files/"
id = 1
questions = []
answers = []
for file in os.listdir(directory):
    with open(directory+file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            if check_numeric(row):
                question_dict = {"quant_cell_positions":gen_quant_cells(row),
                                "processed_question": row[0],
                                "raw_question": row[0],
                                "is_quadratic": False,
                                "Id": id}
                questions.append(question_dict)
                answer_dict = {"answer": gen_answer(row), "Id": id}
                answers.append(answer_dict)
                id += 1


with open('test.json', 'w') as outfile:
    json.dump(questions, outfile)
with open('answers.json', 'w') as outfile:
    json.dump(answers, outfile)
