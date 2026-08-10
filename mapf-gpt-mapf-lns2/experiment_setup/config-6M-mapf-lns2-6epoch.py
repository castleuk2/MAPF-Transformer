# Continue the shared-data 3-epoch MAPF-GPT-6M checkpoint to six total epochs.
out_dir = "runs/mapf_gpt_6m_mapf_lns2_6epoch"
init_from = "resume"
resume_from = "checkpoints_3epoch/mapf_gpt_6m_last_epoch3.pt"

train_data_file = "data/mapf_lns2_same_samples/train"
valid_data_file = "data/mapf_lns2_same_samples/validation"

n_layer = 8
n_head = 8
n_embd = 256
block_size = 256
dropout = 0.0
bias = False

batch_size = 128
gradient_accumulation_steps = 2
max_iters = 81234

learning_rate = 3e-4
min_lr = 3e-5
warmup_iters = 2000
decay_lr = True
lr_decay_iters = 81234
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

eval_interval = 2000
eval_iters = 256
log_interval = 20
always_save_checkpoint = False
compile = False
