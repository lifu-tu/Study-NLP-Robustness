exp=debug
model=cbow
task=SNLI
a-task=PAWS
lr=2e-5
gpu=0
bs=16
interval=5000
test-split=dev
train-split=train
a-train-split=train
a-test-split=dev
num_ex=-1
cheat_rate=-1
mxnet_home=/efs/.mxnet
wdrop=0
drop=0.1
nepochs=10
seed=2
optim=bertadam
remove_cheat=False
test_model=valid_best.params
exp_id=None
learningS=0
pretrain=1
train_from=None
w=1
accm=0

from=None
model_type_a=bert
model_name=book_corpus_wiki_en_uncased

train-bert:
	GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --a-task-name $(a-task)   --pretrain $(pretrain)   --batch-size $(bs) --optimizer bertadam --epochs $(nepochs)  --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --test-split $(test-split) --cheat $(cheat_rate) --max-num-examples $(num_ex) --model-type bert  --model-type-a $(model_type_a)  --model-name $(model_name)   --remove-cheat $(remove_cheat) --seed $(seed) --train-split $(train-split)

train-Mbert:
	GLUE_DIR=data/glue_data  python -m src.main --accumulate $(accm)   --task-name $(task) --a-task-name $(a-task) --train-split $(train-split)    --a-train-split $(a-train-split)  --a-test-split $(a-test-split) --train-from $(train_from) --w $(w)  --pretrain $(pretrain)  --batch-size $(bs) --learningS $(learningS)  --optimizer bertadam --epochs $(nepochs) --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --test-split $(test-split) --cheat $(cheat_rate) --max-num-examples $(num_ex) --model-type bertMul  --model-type-a $(model_type_a)  --model-name $(model_name)  --remove-cheat $(remove_cheat)  --seed $(seed)

test:
	GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --a-task-name $(a-task) --train-split $(train-split)    --a-train-split $(a-train-split)  --a-test-split $(a-test-split)  --mode test      --train-from $(train_from) --w $(w)  --pretrain $(pretrain)  --batch-size $(bs) --learningS $(learningS)  --optimizer bertadam --epochs $(nepochs) --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --init-from  $(from)  --test-from ${test_model}   --test-split $(test-split)  --exp-id  $(exp_id)    --cheat $(cheat_rate) --max-num-examples $(num_ex) --model-type bert --remove-cheat $(remove_cheat)  --seed $(seed)

