from utils import *

"""
Question: I is {I} and F is {F} what is the first value that
            results from applying F to I
Expression: I[0]*F[0]+I[1]*F[1]... (for length of F)

Returns a train_data, test_data, and test_answers
Each is a list that contains dictionaries in the associated formats"""
def return_data(train_id, test_id):
    count = 0
    train_data = []
    test_data = []
    test_answers = []
    for i in [[1, 2, 3, 4, 5], [1, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0], [1, 1, 1], [ 1, 0,  1], [0, 1,  1], [0,  1, 1,  1, 0], [0, 2, 3, 4], [ 1,  1, 1,  1,  1]]:
        for f in [[1, 0, 1], [0, 1], [1, 1, 1], [1, 1], [2, 0], [1], [1], [0], [2, 2], [0, 1, 1], [1, 14], [2, 1], [1, 0, 0], [0, 3, 1], [7, 0.5, 0.5], [0.5], [0.5], [0.25,  0.5], [9, 5], [9, 0,  5]]:
            answer = 0
            formula_lst = []
            for index in range(len(f)):
                answer += i[index] * f[index]
                formula_lst += ["(" + (str(i[index]) if i[index]>= 0 else "(0" + str(i[index])+")") + "*" + (str(i[index]) if f[index]>= 0 else "(0" + str(f[index])+ ")") + ")"]
            formula = "+".join(formula_lst)
            #make sure there are no spaces in the formula

            question = "I is {I} and F is {F} what is the first value that results from applying F to I"
            question = question.format(I = format_list(i), F = format_list(f))
            quant_cell_positions = get_quant_cells(question)

            #divide the produced questions into train and test with a 2:1 ratio
            if count % 3 != 0:
                train_dict = {"expression": formula, "quant_cell_positions": quant_cell_positions, "processed_question": question, "raw_question": question, "is_quadratic": False, "Id": train_id, "Expected": answer}
                train_data.append(train_dict)
                train_id += 1
            else:
                test_dict = {"quant_cell_positions": quant_cell_positions, "processed_question": question, "raw_question": question, "is_quadratic": False, "Id": test_id}
                test_data.append(test_dict)
                answer_dict = {"Id": test_id, "answer": answer, "q_type" : "fil2"}
                test_answers.append(answer_dict)
                test_id += 1
            count += 1
    return train_data, test_data, test_answers
