from collections import Counter
import itertools
import json
import re

import numpy as np

def tokenize_and_separate_quants(data, n_min_vocab):
    pattern = re.compile('\d+,\d+|\d*\.\d+|\d+')
    constant_counts = Counter()
    n_max_nP = 0 # Maximum number of quantities in the input
    for d in data:
        question = d['processed_question'].strip()
        d['in_tokens'] = in_tokens = []
        nP = []
        for token in question.split(' '):
            if pattern.fullmatch(token):
                nP.append(float(token.replace(',', '')))
                in_tokens.append('NUM')
            else:
                in_tokens.append(token)
        nP = np.array(nP)

        n_max_nP = max(len(nP), n_max_nP)

        if 'expression' in d:
            expression = d['expression'] # Ground truth expression
            # Find all quantities in the expression
            out_ops = pattern.split(expression)
            out_quants = map(float, pattern.findall(expression))
            out_tokens = []
            for op, out_quant in itertools.zip_longest(out_ops, out_quants, fillvalue=None):
                out_tokens.extend(op)
                if out_quant is not None: # The last out_quant is None due to zip_longest
                    equals, = np.nonzero(out_quant == nP)
                    if len(equals) == 0: # Output quantity not found in the input. Record quantity as constant
                        constant_counts[out_quant] += 1
                        out_tokens.append(f'{out_quant:g}')
                    else:
                        out_tokens.append(tuple(equals))
            d['out_tokens'] = out_tokens

        d['nP'] = np.array([f'{x:g}' for x in nP])
        d['nP_positions'], = (np.array(in_tokens) == 'NUM').nonzero()
    constants = ['%g' % n for n, count in constant_counts.items() if count >= n_min_vocab]
    return constants, n_max_nP

def infix_to_prefix(expression):
    op_priority = {'+': 0, '-': 0, '*': 1, '/': 1}
    it = iter(reversed(expression)) # Iterate over expression in reverse
    output = []
    def helper():
        op_stack = []
        for c in it:
            if c == '(':
                break
            elif c == ')':
                helper() # When we encounter a closing bracket, use a new stack for things in the parentheses
            elif c in op_priority:
                while op_stack and op_priority[op_stack[-1]] > op_priority[c]:
                    output.append(op_stack.pop())
                op_stack.append(c)
            else:
                output.append(c)
        output.extend(reversed(op_stack))
    helper()
    return output[::-1]

class Vocabulary:
    def __init__(self, words, pad='<pad>', unk='<unk>'):
        self.idx2token = words
        self.token2idx = {w: i for i, w in enumerate(words)}
        self.pad = self.token2idx.get(pad, None)
        self.unk = self.token2idx.get(unk, None)
        self.n = len(words)

def convert_word_to_bytepair_tokenization(d, t5_tokenizer):
    import difflib
    t5_space = '▁'
    d_tokens = d['processed_question'].split(' ')
    question = d['raw_question']

    t_tokens = [x.replace(t5_space, '') for x in t5_tokenizer.tokenize(question)]
    t_tokens = [x for x in t_tokens if x]

    t_join = ''.join(t_tokens)
    d_join = ''.join(d_tokens)
    if t_join == d_join:
        t2d = np.arange(len(t_join)).reshape(-1, 1)
        d2t = np.arange(len(d_join)).reshape(-1, 1)
    else:
        i_t = i_d = 0
        t2d = np.empty((len(t_join),), dtype=object)
        d2t = np.empty((len(d_join),), dtype=object)

        to_add = []
        to_sub = []
        for diff, _, char in difflib.ndiff(t_join, d_join):
            if diff == '+':
                to_add.append(i_d)
                i_d += 1
            elif diff == '-':
                to_sub.append(i_t)
                i_t += 1
            else:
                for i_d_ in to_add:
                    d2t[i_d_] = to_sub
                for i_t_ in to_sub:
                    t2d[i_t_] = to_add
                to_add = []
                to_sub = []

                t2d[i_t] = [i_d]
                d2t[i_d] = [i_t]
                i_t += 1
                i_d += 1
        for i_d_ in to_add:
            d2t[i_d_] = to_sub
        for i_t_ in to_sub:
            t2d[i_t_] = to_add
        assert i_t == len(t_join) and i_d == len(d_join)

    t_pos = np.concatenate([np.full((len(token),), i) for i, token in enumerate(t_tokens)])

    d2t_splits = np.split(d2t, np.cumsum([len(dtok) for dtok in d_tokens])[:-1])
    d_pos_to_t_pos = []
    for i_d, split in enumerate(d2t_splits):
        id_t_pos = set(t_pos[i_t] for i_ts in split for i_t in i_ts)
        d_pos_to_t_pos.append(sorted(id_t_pos))

    # Convert indices
    d['quant_cell_positions'] = [x for qc_pos in d['quant_cell_positions'] for x in d_pos_to_t_pos[qc_pos]]
    d['nP_positions'] = [d_pos_to_t_pos[nP_pos][0] for nP_pos in d['nP_positions']]
    d['in_tokens'] = t_tokens

def setup(use_t5, train_path='data/train.json', test_path='data/test.json', n_min_vocab=5, seed=0, test_split=0.1, do_eval=False):
    with open(train_path, 'r') as f:
        data = json.load(f)
    constants, n_max_nP = tokenize_and_separate_quants(data, n_min_vocab)

    np.random.seed(seed)
    np.random.shuffle(data)
    n_test = int(test_split * len(data))
    train_data, val_data = data[:-n_test], data[-n_test:]

    default_tokens = ['<pad>', '<unk>']
    operation_tokens = ['+', '-', '*', '/']
    if use_t5:
        from transformers import T5Tokenizer, T5Model
        # https://arxiv.org/pdf/1910.10683.pdf
        # https://huggingface.co/transformers/model_doc/t5.html
        # https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_t5.py

        t5_tokenizer = T5Tokenizer.from_pretrained(f't5-{use_t5}')
        t5_model = T5Model.from_pretrained(f't5-{use_t5}')
        in_vocab = Vocabulary([k for k, i in sorted(t5_tokenizer.get_vocab().items(), key=lambda ki: ki[1])], t5_tokenizer.pad_token, t5_tokenizer.unk_token)
    else:
        in_counts = Counter()
        for d in train_data:
            in_counts.update(d['in_tokens'])
        in_vocab = Vocabulary([w for w, c in in_counts.items() if c >= n_min_vocab] + default_tokens)
        t5_model = None
    out_vocab = Vocabulary(operation_tokens + constants + [(i,) for i in range(n_max_nP)] + default_tokens)
    out_vocab.constants = constants
    out_vocab.n_constants = len(constants)
    out_vocab.n_ops = len(operation_tokens)
    out_vocab.base_op = 0
    out_vocab.base_quant = out_vocab.base_constant = out_vocab.base_op + out_vocab.n_ops
    out_vocab.base_nP = out_vocab.base_constant + out_vocab.n_constants

    if do_eval:
        with open(test_path, 'r') as f:
            test_data = json.load(f)
        tokenize_and_separate_quants(test_data, n_min_vocab)
        if use_t5:
            for d in test_data:
                convert_word_to_bytepair_tokenization(d, t5_tokenizer)
        return test_data, in_vocab, out_vocab, n_max_nP, t5_model
    else:
        for d in itertools.chain(train_data, val_data):
            d['out_tokens'] = infix_to_prefix(d['out_tokens'])
            if use_t5:
                convert_word_to_bytepair_tokenization(d, t5_tokenizer)
            d['nP_candidates'] = candidates = {}
            nP = d['nP']
            for j, out_token in enumerate(d['out_tokens']):
                if out_token not in out_vocab.token2idx:
                    # Token is a quantity not in the vocab. Generally this happens for two cases
                    if isinstance(out_token, tuple):
                    # 1. The equation contains two of the same numbers, e.g. ['+', '8', '8'], and we don't know which number comes first in the English sentence
                        candidates[j] = np.array(out_token)
                    else:
                    # 2. The equation contains a number which represents English words such as 'nickel', 'dime', 'quarter', 'eight', etc. which was not common enough to translate into a constant during parsing. There's not much we can do here
                        candidates[j] = np.arange(len(nP))
        train_data = [d for d in train_data if not d['is_quadratic']]
        return train_data, val_data, in_vocab, out_vocab, n_max_nP, t5_model

def evaluate_prefix_expression(expression):
    operators = {'+': np.add, '-': np.subtract, '*': np.multiply, '/': np.divide}
    def helper(start):
        if start == len(expression):
            print('Malformed expression', expression)
            return np.nan, start
        if expression[start] not in operators:
            return float(expression[start]), start + 1
        fn = operators[expression[start]]
        arg1, end = helper(start + 1)
        arg2, end = helper(end)
        return fn(arg1, arg2), end
    result, end = helper(0)
    if end != len(expression):
        print('Malformed expression', expression)
        return np.nan
    return result

def check_match(pred, d):
    ref = d['out_idxs']
    match_expression = pred == ref
    if not match_expression:
        nP = d['nP']
        pred = sub_nP(d['pred_tokens'], nP)
        ref = sub_nP(d['out_tokens'], nP)
        match_expression = pred == ref
    return match_expression or abs(evaluate_prefix_expression(pred) - evaluate_prefix_expression(ref)) < 1e-4, match_expression

def sub_nP(tokens, nP):
    return [nP[t[0]] if isinstance(t, tuple) else t for t in tokens]
