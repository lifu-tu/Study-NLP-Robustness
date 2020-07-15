# An Empirical Study on Robustness to Spurious Correlations using Pre-trained Language Models

Code for the paper "[An Empirical Study on Robustness to Spurious Correlations using Pre-trained Language Models](https://arxiv.org/abs/2007.06778)", accept by TACL 2020. Part of the code is from "[here](https://github.com/hhexiy/debiased)".


## Dependencies
- Python 3.6
- [MXNet 1.6.0](https://mxnet.apache.org/get_started/?platform=linux&language=python&processor=gpu&environ=pip&), e.g., using cuda-10.0, `pip install mxnet-cu100`
- [GluonNLP 0.9.0](https://github.com/dmlc/gluon-nlp/)


## Following are several important commands:

## Train models


### single task learning

**one exmple for NLI task training on MNLI dataset(BERT-BASE)**
```
make train-bert exp=mnli_seed/bert task=MNLI   test-split=dev_matched bs=32   gpu=0 nepochs=10  seed=2 lr=0.00002
```

**one exmple for para task training on QQP dataset (RoBERTa-BASE)**
```
make train-bert exp=qqp_seed/roberta task=QQP   test-split=dev  gpu=3  nepochs=10  model_type_a=roberta   model_name=openwebtext_ccnews_stories_books_cased bs=32  seed=2  lr=0.00002
```

**one example for NLI task training on MNLI dataset with Roberta large**
```
make train-bert exp=mnli_seed/robertal task=MNLI   test-split=dev_matched  gpu=0 nepochs=10  model_type_a=robertal   model_name=openwebtext_ccnews_stories_books_cased   seed=2 lr=0.00002
```

### multi-task learning (MTL)



**one example for BERT multi-task learning (MNLI and QQP):** 
```
make train-Mbert exp=mnli_seed_m/ber task=MNLI  a-task=QQP   test-split=dev_matched  model_type_a=bert   gpu=0 nepochs=10  seed=2 learningS=1 lr=0.00002
```

**one example for RoBERTa multi-task learning (MNLI and QQP):**

```
make train-Mbert exp=mnli_seed_m/ber task=MNLI  a-task=QQP   test-split=dev_matched  model_type_a=roberta  model_name=openwebtext_ccnews_stories_books_cased  gpu=0 nepochs=10  seed=2 learningS=1 lr=0.00002
```

**one example for RoBERTa-Larger multi-task learning with gradient accumulation (MNLI and QQP):**

```
make train-Mbert exp=mnli_seed_m/robertal task=MNLI  a-task=PAWSall  train-split=mnli_snli_train   a-train-split=paws_qqp   test-split=dev_matched  bs=4 accm=8  model_type_a=robertal model_name=openwebtext_ccnews_stories_books_cased gpu=2 nepochs=5  seed=2 learningS=0 lr=0.00002
```


## Evaluation:
Following are several examples for the evaluation of trained models

- SNLI: `make test test-split=test from=[path to model] test_model=[model]   task=SNLI`
- HANS: `make test test-split=lexical_overlap   from=[path to model]  test_model=[model]    task=MNLI-hans`
- PAWS: `make test test-split=dev   from=[path to model]  test_model=[model]  task=PAWS` 


## References
```
@article{tu20tacl,
    title = {An Empirical Study on Robustness to Spurious Correlations using Pre-trained Language Models
},
    author = {Lifu Tu and Garima Lalwani and Spandana Gella and He He},
    journal = {Transactions of the Association of Computational Linguistics},
    month = {},
    url = {https://arxiv.org/abs/2007.06778},
    year = {2020}
}
```

