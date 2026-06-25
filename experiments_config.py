import pickle
import os, sys
root = os.path.dirname(os.getcwd())
path = os.path.join(root,"EEC289A-unsupervised-learning/notebooks")

"""
Random_0:
"""

PAD_TOKEN = 50303
BOS_TOKEN = 50302
CONTEXT_LEN = 128
STRIDE = 64
BATCH_SIZE = 128
TRAIN_SUBSET = 100000 # reference: 100000
CURRICULUM = 0 # 0: random, -1: anti-curriculum, 1: curriculum
SCORE = "unigram" # "unigram" or "bigram"
TAG = "random_0"

######################################################################

"""
Curriculum_0:
"""

PAD_TOKEN = 50303
BOS_TOKEN = 50302
CONTEXT_LEN = 128
STRIDE = 64
BATCH_SIZE = 128
TRAIN_SUBSET = 100000 # reference: 100000
CURRICULUM = 1 # 0: random, -1: anti-curriculum, 1: curriculum
SCORE = "unigram" # "unigram" or "bigram"
TAG = "curriculum_0"


######################################################################

"""0:"""
MODEL_KWARGS = dict(
    vocab_size=50304,
    block_size=CONTEXT_LEN,
    n_layer=2,
    n_embd=256,
    n_head=4,
    bias=False,
    dropout=0.1,
    pad_token=PAD_TOKEN
)
TRAINER_KWARGS = dict(
    epochs=100,
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    grad_norm_clip=1.0,
    model_tag=TAG
)

try:
    with open(os.path.join(path, "history_random_0"), "rb") as file:  # 'wb' = write binary
        history = pickle.load(file)

    print("Dictionary loaded successfully.")
except (OSError, pickle.PickleError) as e:
    print(f"Error loading dictionary: {e}")

print(history["train_loss"])
print(history["test_nll"])


