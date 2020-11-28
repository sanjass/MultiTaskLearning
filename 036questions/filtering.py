#if I = {I} and F = {F} what is the length of applying F to I HINT use subtraction

def format_list(lst):
    string = "[ "
    string += " ".join(map(lambda x: str(x), lst))
    string += " ] "
    return string


def return_data(train_id, test_id):
#can try this with just all of the positions? havent yet
    # quant_cell_positions = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 23, 24, 25, 26, 27, 28, 29, 30]
    count = 0
    train_data = []
    test_data = []
    test_answers = []
    for i in [[1, 2, 3, 4, 5], [1, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0]]:
        for f in [[-1, 0, 1], [0, 1], [1, 1, 1], [-1, -1], [2, 0]]:
            answer = len(i)-len(f)+1

            #make sure there are no spaces in the formula
            formula = "{len_I}-{len_F}+1"
            formula = formula.format(len_I = len(i), len_F = len(f))
            question = "I has length {len_I} and F has length {len_F} what is the length of the result of applying F to I"
            question = question.format(I = format_list(i), F = format_list(f), len_I = len(i), len_F = len(f))
            quant_cell_positions = get_quant_cells(question)

            if count % 3 != 0:
                train_dict = {"expression": formula, "quant_cell_positions": quant_cell_positions, "processed_question": question, "raw_question": question, "is_quadratic": False, "Id": train_id, "Expected": answer}
                train_data.append(train_dict)
                train_id += 1
            else:
                test_dict = {"quant_cell_positions": quant_cell_positions, "processed_question": question, "raw_question": question, "is_quadratic": False, "Id": test_id}
                test_data.append(test_dict)
                answer_dict = {"Id": test_id, "answer": answer}
                test_answers.append(answer_dict)
                test_id += 1
            count += 1
    return train_data, test_data, test_answers

def get_quant_cells(question):
    q_split = question.split(" ")
    return [i for i in range(len(q_split))]
