"""Returns a list of numbers from 0 to number of words in the question
to be used as quant cells for data"""
def get_quant_cells(question):
    return [i for i in range(len(question.split(" ")))]

"""Formats negative numbers to use the word negative instead of '-'"""
def format_num(x):
    return  "negative " + str(-1*x) if x<0  else str(x)

"""Replaces a negative number with 0-num for expressions"""
def format_exp(x):
    return  "0-" + str(-1*x) if x<0  else str(x)

"""Formats a list of numbers to be entered with appropriate negative signs"""
def format_list(lst):
    string = "[ "
    string += " ".join(map(lambda x: "negative " + str(-1*x) if x<0  else str(x), lst))
    string += " ] "
    return string

def get_answer(s0, w, x):
    s = s0
    for _x in x:
        s = s*w + _x
    return s

def get_expression(s0, w, x):
    expression_str = ""
    expression_str = str(s0)
    for _x in x:
        expression_str = "((" + expression_str + "*" + str(w) + ")" + "+" + str(_x) + ")"
    return expression_str
