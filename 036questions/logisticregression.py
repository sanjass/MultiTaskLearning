#x = [1, 1], theta = [1, 0], theta_0 = -0.5 and y = 0.
#what is the value of theta * x + theta_0?

#messy code atm will clean up
#if you remove the negative signs does well
#is iffy with the negative will update if its a formatting issue
# import json

def return_data(train_id, test_id):
#can try this with just all of the positions? havent yet
    # quant_cell_positions = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 23, 24, 25, 26, 27, 28, 29, 30]
    count = 0
    train_data = []
    test_data = []
    test_answers = []
    for theta_a, theta_b in [(1, 1), (0, 0), (-1, 0), (0, 1), (1, 2), (2, 1), (8, 4), (4, 5), (3, 4)]:
        for theta_0 in [0.5, 0, 0.25, 0.25, 1, -6, 6, 18, 3]:
            for x_a, x_b in [(1, 1), (0, 0), (1, 0), (0, 1), (-1, 0)]:

                answer = (x_a * theta_a + x_b * theta_b) + theta_0

                #make sure there are no spaces in the formula
                formula = "(({x_a}*{theta_a})+({x_b}*{theta_b}))+{theta_0}"
                formula = formula.format(x_a = x_a, x_b = x_b, theta_a = theta_a, theta_b = theta_b, theta_0 = theta_0)
                question = "x = ( {x_a} , {x_b} ) , theta = ( {theta_a} , {theta_b} ) and theta_0 = {theta_0} . What is the value of theta times x plus theta_0 ?"
                question = question.format(x_a = x_a, x_b = x_b, theta_a = theta_a, theta_b = theta_b, theta_0 = theta_0)
                quant_cell_positions = get_quant_cells(question)

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

def get_quant_cells(question):
    return [i for i in range(len(question.split(" ")))]
