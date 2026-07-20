# Fair scratch-training comparison with the 8.81M MAPF Transformer.
out_dir = "runs/mapf_gpt_6m_mapf_lns2"
init_from = "scratch"

train_data_file = "data/mapf_lns2_same_samples/train"
valid_data_file = "data/mapf_lns2_same_samples/validation"

# Official MAPF-GPT-6M architecture (about 6.31M non-position parameters).
n_layer = 8
n_head = 8
n_embd = 256
block_size = 256
dropout = 0.0
bias = False

# Two GPUs: 128 samples/GPU * one local micro-step = global batch 256.
batch_size = 128
gradient_accumulation_steps = 2
max_iters = 40617

learning_rate = 3e-4
min_lr = 3e-5
warmup_iters = 2000
decay_lr = True
lr_decay_iters = 40617
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

eval_interval = 2000
eval_iters = 256
log_interval = 20
always_save_checkpoint = False
compile = False
