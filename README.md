# An Empirical Study on Robustness to Spurious Correlations using Pre-trained Language Models

Code for the paper "[An Empirical Study on Robustness to Spurious Correlations using Pre-trained Language Models](https://arxiv.org/abs/2007.06778)", accept by TACL 2020. Part of the code is from "[here](https://github.com/hhexiy/debiased)".


## Dependencies
- Python 3.6
- [MXNet 1.6.0](https://mxnet.apache.org/get_started/?platform=linux&language=python&processor=gpu&environ=pip&), e.g., using cuda-10.0, `pip install mxnet-cu100`
- [GluonNLP 0.9.0](https://github.com/dmlc/gluon-nlp/)


## Following are several important commands:

## Train models

### Baseline (finetuning for 3 epochs)
#### train on MNLI, test on MNLI dev and HANS
```
make train-bert exp=mnli_seed/bert task=MNLI   test-split=dev_matched bs=32   gpu=0 \
     nepochs=3  seed=2 lr=0.00002
```
#### train on QQP, test on QQP dev and PAWS
```
make train-bert exp=mnli_seed/bert task=QQP   test-split=dev bs=32   gpu=0  \
     nepochs=3  seed=2 lr=0.00002
```

- `exp`: the directory to save models
- `task`: which dataset to load.
- `test-split`: which split to use for validation
- `bs`: batch size
- `gpu`: which gpu to use
- `nepochs`: the number of finetuning epoch
- `seed`: random seed number
- `lr`: learning rate


### Single task learning (finetuning for more epochs)

#### Training on MNLI dataset (BERT-BASE)
```
make train-bert exp=mnli_seed/bert task=MNLI   test-split=dev_matched bs=32 \
   gpu=0 nepochs=10  seed=2 lr=0.00002
```

#### Training on QQP dataset (RoBERTa-BASE)
```
make train-bert exp=qqp_seed/roberta task=QQP   test-split=dev  gpu=3 \
     nepochs=10 model_type_a=roberta  model_name=openwebtext_ccnews_stories_books_cased \
     bs=32  seed=2  lr=0.00002
```

#### Training on MNLI dataset with Roberta large
```
make train-bert exp=mnli_seed/robertal task=MNLI   test-split=dev_matched  \
     gpu=0 nepochs=10 model_type_a=robertal   model_name=openwebtext_ccnews_stories_books_cased \
     seed=2 lr=0.00002
```

- `model_type_a`: which pretrained language are used: 'bert': BERT; 'bertl': BERT LARGE; 'roberta': RoBERTa; 'robertal': RoBERTa LARGE.
- `model_name`: the dataset used for language model training: 'book\_corpus\_wiki\_en\_uncased' for BERT, 'openwebtext\_ccnews\_stories\_books\_cased' foor RoBERTa


### Multi-task learning (MTL)

#### BERT multi-task learning on MNLI and QQP: 
```
make train-Mbert exp=mnli_seed_m/ber task=MNLI  a-task=QQP   test-split=dev_matched \
     model_type_a=bert gpu=0 nepochs=10  seed=2 learningS=1 lr=0.00002
```

#### RoBERTa multi-task learning on MNLI and QQP:
```
make train-Mbert exp=mnli_seed_m/ber task=MNLI  a-task=QQP   test-split=dev_matched \
     model_type_a=roberta model_name=openwebtext_ccnews_stories_books_cased \
     gpu=0 nepochs=10  seed=2 learningS=1 lr=0.00002
```

#### RoBERTa-Large multi-task learning one MNLI and QQP with gradient accumulation:
```
make train-Mbert exp=mnli_seed_m/robertal task=MNLI  a-task=PAWSall  train-split=mnli_snli_train \
      a-train-split=paws_qqp test-split=dev_matched  bs=4 accm=8  model_type_a=robertal \
      model_name=openwebtext_ccnews_stories_books_cased gpu=2 nepochs=5 \
      seed=2 learningS=0 lr=0.00002
```
- `task`: the target datasets
- `a-task`: the auxiliary datasets
- `learningS`: 0:gradient accumulation; 1: traditional MTL tasking
- `accm` : the number of steps for gradient accumulation
- `train-split` : which split to use for training
- `a-train-split` : which split to use for training for auxiliary datasets


## Evaluation:
Following are several examples for the evaluation of trained models on the specific task:

#### SNLI: 
`make test test-split=test from=[path to model] test_model=[model]   task=SNLI`
#### HANS: 
`make test test-split=lexical_overlap   from=[path to model]  test_model=[model]    task=MNLI-hans`
#### PAWS: 
`make test test-split=dev   from=[path to model]  test_model=[model]  task=PAWS` 

- `test-split` : which split to be evaluated
- `from` : the directory to save models
- `test_model` : the save model file name
- `task` :  which dataset to be evaluated


## How to try your own datasets?
In the file `dataset.py`, you can implement your own dataset class (please see several examles in the file). 
Then add your dataset class in the file `task.py`. Now you can set parameters on your task for training.
For example, if the dataset is called `XXX`, 
```
make train-Mbert exp=mnli_seed_m/ber task=MNLI  a-task=XXX   test-split=dev_matched \
     model_type_a=bert gpu=0 nepochs=10  seed=2 learningS=1 lr=0.00002
``` 
The above examples is to finetune BERT on MNLI and XXX, and do early stopping on MNLI `dev_mached` split.


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

