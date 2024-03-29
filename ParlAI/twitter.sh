#! /usr/bin/bash

# Set-up the environment.
NUM_EPOCHS=10
MODEL_FILE=trained/twitter/s2s
DT=train
BATCH_SIZE=256
TRUNCATE=20
HIDDEN_SIZE=512
EMBED_SIZE=300
NUM_LAYERS=2
DROP=0.1
LR=0.001
LR2=0.001
CLIP=1
LT=enc_dec
ATT=general
NUM_WORDS=30000
VMT=accuracy
VMM=max
OPT=adam
OPT2=adam

mkdir -p `dirname $MODEL_FILE`

# Start the experiment.
python3 -u examples/train_model.py -opt2 $OPT2 -lr2 $LR2 -m seq2seq_custom -t twitter -mf $MODEL_FILE -dt $DT -opt $OPT -hs $HIDDEN_SIZE -esz $EMBED_SIZE -nl $NUM_LAYERS -att $ATT -dr $DROP -lr $LR -clip $CLIP -lt $LT -vmt $VMT -vmm $VMM -vtim 3600 -tok nltk -bs $BATCH_SIZE -tr $TRUNCATE --batch-sort false --dict-maxtokens $NUM_WORDS --dict-lower True -eps $NUM_EPOCHS
