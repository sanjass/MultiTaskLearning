from utils import *

"""
Question: x = [1, 1], theta = [1, 0], theta_0 = -0.5 and y = 0.
            what is the value of theta * x + theta_0?
Expression: (({x_a}*{theta_a})+({x_b}*{theta_b}))+{theta_0}

Returns a train_data, test_data, and test_answers
Each is a list that contains dictionaries in the associated formats"""
def return_data(train_id, test_id):
    count = 0
    train_data = []
    test_data = []
    test_answers = []
    for theta_a, theta_b in [(1, -1), (0, 0), ( 1, 0), (0, 1), (1, -2)]:
        for theta_0 in [0.5, 0, 0.25, -1,  6, 6, 18, -3]:
            for x_a, x_b in [(1, -1), (0, 0), (1, 0), (0, -1), (-1, 0)]:
                answer = (x_a * theta_a + x_b * theta_b) + theta_0

                #make sure there are no spaces in the formula
                formula = "(({x_a}*{theta_a})+({x_b}*{theta_b}))+{theta_0}"
                formula = formula.format(x_a = format_exp(x_a), x_b = format_exp(x_b), theta_a = format_exp(theta_a), theta_b = format_exp(theta_b), theta_0 = format_exp(theta_0))
                question = "x is ( {x_a} , {x_b} ) , theta is ( {theta_a} , {theta_b} ) and theta_0 is {theta_0} . What is the value of theta times x plus theta_0 ?"
                question = question.format(x_a = format_num(x_a), x_b = format_num(x_b), theta_a = format_num(theta_a), theta_b = format_num(theta_b), theta_0 = format_num(theta_0))
                quant_cell_positions = get_quant_cells(question)

                #divide the produced questions into train and test with a 2:1 ratio
                if count % 3 != 0:
                    train_dict = {"expression": formula, "quant_cell_positions": quant_cell_positions, "processed_question": question, "raw_question": question, "is_quadratic": False, "Id": train_id, "Expected": answer}
                    train_data.append(train_dict)
                    train_id += 1
                else:
                    test_dict = {"quant_cell_positions": quant_cell_positions, "processed_question": question, "raw_question": question, "is_quadratic": False, "Id": test_id}
                    test_data.append(test_dict)
                    answer_dict = {"Id": test_id, "answer": answer , "q_type" : "logreg"}
                    test_answers.append(answer_dict)
                    test_id += 1
                count += 1
    return train_data, test_data, test_answers
