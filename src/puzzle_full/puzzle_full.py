import sys
import os
import importlib
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset

torch.manual_seed(1337)  #pytorch seed
np.random.seed(1337) #numpy seed
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337) #main GPU seed 
    torch.cuda.manual_seed_all(1337) #multi-GPU seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from dataloader import ChessIterableDataset, pad_collate
from dataclass import VariableRunConfig, DataConfig, HyperParamConfig, RunConfig, ChessConfig, PuzzleConfig, PretrainConfig
from auxilliary import retrieve_iteration_number, write_to_file
from uci_move_dict import MoveDictionary

sys.path.append('../models/model1')
import chess_model
importlib.reload(chess_model)  # Reloads the module
from chess_model import Chess  # Now import the class

db_path = '/workspace/database/lichess/puzzle.db'

debug_path = "debug.txt"
model_iteration = PretrainConfig.iter
existing_model_path = f"../runs/lichess_run/iters/state_dict_v{model_iteration}.pth"

iteration = retrieve_iteration_number(model_iteration, PuzzleConfig.min_rating)
print(f"{iteration=}")
log_path = f"/workspace/src/puzzle_tests/lichess/model_{iteration}.txt"
if log_path is not None:
    with open(log_path, 'w') as file:
        file.write(f"model_iter: {model_iteration}\n")
        file.write(f"type: puzzle_full\n")

move_dict = MoveDictionary()
index_to_move = move_dict.index_move_dict
move_to_index = move_dict.move_index_dict


torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Chess(ChessConfig(), existing_model_path, device)
model.to(device)
print(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters: ", total_params)
# If your policy head parameters are scattered in the model
policy_params = sum(p.numel() for name, p in model.named_parameters() if 'policy' in name and p.requires_grad)
print(f"Total number of parameters in the policy head: {policy_params}")

model = torch.compile(model)



total_batch_size = RunConfig.total_batch_size # used for alphazero
batch_size = VariableRunConfig.gpu_batch_size
assert total_batch_size % batch_size == 0
grad_accum_steps = total_batch_size // batch_size

optimizer = model.configure_optimizer(weight_decay=RunConfig.adamw_weight_decay, learning_rate=HyperParamConfig.constant_lr, device=device)

def puzzle_accuracy_single_move(model, puzzle_loader, device, RunConfig, log_path, masking, puzzle_complexity):
    model.eval()
    accuracy_list = []
    val_iter = iter(puzzle_loader)
    print("starting accuracy validation")
    with torch.no_grad():
        gpu_step = 0
        while True:
            try:
                board_state_tensor, special_token_tensor, target_p_tensor, legal_moves_tensor = next(val_iter)
                
            except StopIteration:
                break
            board_state_tensor, special_token_tensor, target_p_tensor = board_state_tensor.to(device), special_token_tensor.to(device), target_p_tensor.to(device)
            if masking:
                legal_moves_tensor = legal_moves_tensor.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                batch_accuracy = model.puzzle_accuracy(board_state_tensor, special_token_tensor, target_p_tensor, legal_moves_tensor)
            print(f"gpu_step:{gpu_step}, puzzle accuracy={batch_accuracy}, complexity={puzzle_complexity}")
            gpu_step += 1
            accuracy_list.append(batch_accuracy.item())
        loss_accum = sum(accuracy_list)/len(accuracy_list)

    if log_path is not None:
        with open(log_path, 'a') as log_file:
            log_file.write(f"Validation puzzle accuracy={loss_accum}, for complexity={puzzle_complexity}\n")

    print(f"Validation puzzle accuracy={loss_accum}")

for i in range(15):
    puzzle_dataset = ChessIterableDataset(db_path, VariableRunConfig.n_limit, VariableRunConfig.token_encoding_scheme, (i+1)*2, index_to_move, move_to_index, VariableRunConfig.masking, min_rating=PuzzleConfig.min_rating)
    puzzle_loader = DataLoader(puzzle_dataset, batch_size=VariableRunConfig.gpu_batch_size, num_workers=DataConfig.n_workers, collate_fn=pad_collate)

    puzzle_accuracy_single_move(model, puzzle_loader, device, RunConfig, log_path, VariableRunConfig.masking, (i+1)*2)


# if __name__ == '__main__':
#     main()

