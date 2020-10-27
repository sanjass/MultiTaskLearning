Code taken from https://github.com/hendrycks/test
# Measuring Massive Multitask Language Understanding
This is the repository for [Measuring Massive Multitask Language Understanding](https://arxiv.org/pdf/2009.03300) by
[Dan Hendrycks](https://people.eecs.berkeley.edu/~hendrycks/), [Collin Burns](http://collinpburns.com), [Steven Basart](https://stevenbas.art), Andy Zou, Mantas Mazeika, [Dawn Song](https://people.eecs.berkeley.edu/~dawnsong/), and [Jacob Steinhardt](https://www.stat.berkeley.edu/~jsteinhardt/).

This repository contains OpenAI API evaluation code, and the test is available for download [here](https://people.eecs.berkeley.edu/~hendrycks/data.tar).

## Test Leaderboard

If you want to have your model added to the leaderboard, please reach out to us or submit a pull request.


Results of the test:
|                Model               | Authors |  Humanities |  Social Science  | STEM | Other | Average |
|------------------------------------|----------|:-------:|:-------:|:-------:|:-------:|:-------:|
| [GPT-3](https://arxiv.org/abs/2005.14165) | Brown et al., 2020 | 40.8 | 50.4 | 36.7 | 48.8 | 43.9
| [UnifiedQA](https://arxiv.org/abs/2005.00700) | Khashabi et al., 2020 | 45.6 | 56.6 | 40.2 | 54.6 | 48.9
| Random Baseline           | N/A | 25.0 | 25.0 | 25.0 | 25.0 | 25.0 | 25.0


## Citation

If you find this useful in your research, please consider citing the test and also the [ETHICS](https://arxiv.org/abs/2008.02275) dataset it draws from:

    @article{hendryckstest2020,
      title={Measuring Massive Multitask Language Understanding},
      author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
      journal={arXiv preprint arXiv:2009.03300},
      year={2020}
    }

    @article{hendrycksethics2020,
      title={Aligning AI With Shared Human Values},
      author={Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and Dawn Song and Jacob Steinhardt},
      journal={arXiv preprint arXiv:2008.02275},
      year={2020}
    }
