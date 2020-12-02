from utils import *

"""
Question: Consider a very simple RNN, defined by the following equation:
            s_t = w*s_t-1 + x_t. Given s_0 = 0 and w=1 and x = [1,1,1],
            what is s_3?
Expression: w*s_t-1 + x_t (recursively)

Returns a train_data, test_data, and test_answers
Each is a list that contains dictionaries in the associated formats"""
def return_data(train_id, test_id):
    count = 0
    train_data = []
    test_data = []
    test_answers = []

    for s0 in [0,1,2,3,1.5]:
        for w in [0, .5, 1, 1.5, .1]:
            for x in [[1,1,1], [1,2,3], [1,0,1], [1,1,1,1], [1,2], [2], [5,2,0,1], [0,0,1]]:
                raw_q = "Consider a very simple RNN, defined by the following equation: s_t = w*s_t-1 + x_t. "
                raw_q += "Given s_0 = " + str(s0)
                raw_q += " and w = " + str(w)
                raw_q += " and x = " + str(x)
                raw_q += " what is s_" + str(len(x)) + "?"

                ans = get_answer(s0, w, x)
                expression = get_expression(s0, w, x)
                quant_cells = get_quant_cells(raw_q)

                #divide the produced questions into train and test with a 2:1 ratio
                if count % 3 != 0:
                    train_dict = {"expression": expression, "quant_cell_positions": quant_cells, "processed_question": raw_q, "raw_question": raw_q, "is_quadratic": False, "Id": train_id, "Expected": ans}
                    train_data.append(train_dict)
                    train_id += 1
                else:
                    test_dict = {"quant_cell_positions": quant_cells, "processed_question": raw_q, "raw_question": raw_q, "is_quadratic": False, "Id": test_id}
                    test_data.append(test_dict)
                    answer_dict = {"Id": test_id, "answer": ans, "q_type" : "rnn"}
                    test_answers.append(answer_dict)
                    test_id += 1
                count += 1

    return train_data, test_data, test_answers
