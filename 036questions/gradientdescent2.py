from utils import *
"""
Question: f(theta) is ({c1} times theta plus {c2}) squared and
            theta is {theta} and eta is {eta} what is theta after gd ?
            HINT use 2 times {c1} times theta plus 2 times {c2}
Expression: {theta}-{eta}*2*{c1}*({c1}*{theta}+{c2})

Returns a train_data, test_data, and test_answers
Each is a list that contains dictionaries in the associated formats"""
def return_data(train_id, test_id):
    count = 0
    train_data = []
    test_data = []
    test_answers = []
    for eta in [0.1, 0.11, 0.56, 0.05]:
        for theta in [1, 4,  6, 9.4, 0.23]:
            for c1 in [-3, 4, 7, -10, 0.3]:
                for c2 in [3,  5]:
                    answer = theta - eta*2*c1*(c1*theta+c2)
                    #make sure there are no spaces in the formula
                    formula = "{theta}-{eta}*2*{c1}*({c1}*{theta}+{c2})"
                    formula = formula.format(c1 = format_exp(c1), c2 = c2, theta = theta, eta=eta)
                    question = "f(theta) is ({c1} times theta plus {c2}) squared and theta is {theta} and eta is {eta} what is theta after gd ? HINT use 2 times {c1} times theta plus 2 times {c2}"
                    question = question.format(c1 = format_num(c1), c2 = c2, theta = theta, eta=eta)
                    quant_cell_positions = get_quant_cells(question)

                    #divide the produced questions into train and test with a 2:1 ratio 
                    if count % 3 != 0:
                        train_dict = {"expression": formula, "quant_cell_positions": quant_cell_positions, "processed_question": question, "raw_question": question, "is_quadratic": False, "Id": train_id, "Expected": answer}
                        train_data.append(train_dict)
                        train_id += 1
                    else:
                        test_dict = {"quant_cell_positions": quant_cell_positions, "processed_question": question, "raw_question": question, "is_quadratic": False, "Id": test_id}
                        test_data.append(test_dict)
                        answer_dict = {"Id": test_id, "answer": answer , "q_type" : "gd2"}
                        test_answers.append(answer_dict)
                        test_id += 1
                    count += 1
    return train_data, test_data, test_answers
