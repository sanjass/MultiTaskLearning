# Multitask Learning

Final project for 6.883, Meta Learning course at MIT offered in Fall 2020.

In our work we use the new Multitask Learning benchmark to evaluate performance on popular nlp models in the k-shot learning setting, where k can be 0 or larger. The benchmark contains multiple-choice questions for 57 different subjects.
The benchmark is proposed in [Measuring Massive Multitask Language Understanding](https://arxiv.org/pdf/2009.03300) by
[Dan Hendrycks](https://people.eecs.berkeley.edu/~hendrycks/), [Collin Burns](http://collinpburns.com), [Steven Basart](https://stevenbas.art), Andy Zou, Mantas Mazeika, [Dawn Song](https://people.eecs.berkeley.edu/~dawnsong/), and [Jacob Steinhardt](https://www.stat.berkeley.edu/~jsteinhardt/).
Code taken from https://github.com/hendrycks/test

Parts of the code are inspired by the official OpenAI API evaluation code, [available here](https://github.com/hendrycks/test)

## Data
The test data is available for download [here](https://people.eecs.berkeley.edu/~hendrycks/data.tar).

## Install

1. Clone repo:
 ```
 git clone git@github.com:sanjass/MultiTaskLearning.git
```
2. Install packages from conda env:
```
conda env create -f env.yml
```
3. Activate environment:
 ```
 conda activate pytorch
 ```

## Codebase structure
- `QA_huggingface.ipynb` - This is a jupyter notebook containing the code related to our QA approach with huggingface
- `unifiedQA` - contains code necessary to do the experiments with T5 transformer. The pretrained models can be accessed on the [official unifiedQA repo](https://github.com/allenai/unifiedqa)
- `ex2_data_parsing` - takes our original data and converts the numeric questions into a format that can be used by T5 transformer presented in exercise 2. 
- `ex2_for_final_project.ipynb` - This is a jupyter notebook containing code related to our T5 + GNN based on ex2. This code is mainly used to perform the experiments with our 6.036 dataset. We used google colab to run the notebook. Note that you would need to upload he `util.py` file from `exercise2` as well as the generated `train.json` and `test.json`. See "6.036 Question Generation" section for details on how to generate this dataset.

## 6.036 Question Generation
To generate the train and test data navigate to the 036questions folder and run 'python3 writedata.py'.  This will generate the questions based on the other files in the directory.  To make changes to the generated questions change the numbers in these files and rerun the command.
