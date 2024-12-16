import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import sqlite3
import pandas as pd
from itertools import islice
import numpy as np
import joblib
from torch.utils.data import Dataset, DataLoader, IterableDataset
import json
from torch.distributed import init_process_group, destroy_process_group
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
from torch.nn.utils.rnn import pad_sequence

torch.manual_seed(1337)  #pytorch seed
np.random.seed(1337) #numpy seed
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337) #main GPU seed 
    torch.cuda.manual_seed_all(1337) #multi-GPU seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from dataclass import VariableRunConfig


class ChessIterableDataset(IterableDataset):
    def __init__(self, db_path, split, n_limit, token_encoding_scheme, index_to_move, move_to_index, n1=0.8, n2=0.1, masking=False, pretrain_offset=0):
        self.db_path = db_path
        self.split = split
        self.n1 = n1  # Proportion of training data
        self.n2 = n2  # Proportion of validation data
        self.n_limit = n_limit  # Optional limit on the number of data points
        self.masking = masking
        self.masking_query = ", legal_moves" if masking else ""
        self.pretrain_offset = pretrain_offset
        self.token_encoding_scheme = token_encoding_scheme
        self.index_to_move = index_to_move
        self.move_to_index = move_to_index

    def __iter__(self):
        return self.data_generator()

    def data_generator(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Calculate limits for each split
        total_query = "SELECT COUNT(*) FROM chess_analysis;"
        cursor.execute(total_query)
        total_rows = cursor.fetchone()[0]
        
        if self.n_limit is not None:
            total_rows = min(total_rows, self.n_limit)  # Adjust total_rows based on n_limit

        train_limit = int(total_rows * self.n1)
        val_limit = int(total_rows * self.n2)
        test_limit = total_rows - train_limit - val_limit

        # Prepare the query based on the split
        if self.split == 'train':
            query = f"SELECT board_state, special_tokens, next_move, turn{self.masking_query} FROM chess_analysis LIMIT {train_limit + self.pretrain_offset};"
        elif self.split == 'val':
            query = f"SELECT board_state, special_tokens, next_move, turn{self.masking_query} FROM chess_analysis LIMIT {val_limit} OFFSET {train_limit};"
        elif self.split == 'test':
            query = f"SELECT board_state, special_tokens, next_move, turn{self.masking_query} FROM chess_analysis LIMIT {test_limit} OFFSET {train_limit + val_limit};"

        cursor.execute(query)
        for row in cursor.fetchall():
            board_state = json.loads(row[0])
            special_tokens = json.loads(row[1])
            turn = row[3]
            
            turn_encoding = 1 if turn == "b" else 0
            if self.token_encoding_scheme % 2 == 0: #if encoding 2 or 4: remove turn token
                special_tokens = special_tokens[:-1]
                if turn_encoding == 1: # if blacks turn (and encoding 2 or 4): flip board, piece colour swap, castling rights swap
                    for i, value in enumerate(board_state):
                        if 0 < value < 7: #0,1 for CLS, empty sq
                            board_state[i] += 6
                        elif value >= 7:
                            board_state[i] -= 6
                    new_board_state = []
                    for i in range(7, -1, -1):
                        new_board_state.extend(board_state[i*8:(i+1)*8])
                    board_state = new_board_state
                    temp_0, temp_1 = special_tokens[0], special_tokens[1]
                    special_tokens[0], special_tokens[1] = special_tokens[2], special_tokens[3]
                    special_tokens[2], special_tokens[3] = temp_0, temp_1
            
            
            target_move_index = row[2]
            
            
            if turn_encoding == 1 and self.token_encoding_scheme % 2 == 0:
                target_move_index = self.flip_uci(target_move_index)
            board_state_tensor, special_token_tensor, target_move_tensor = torch.tensor(board_state, dtype=torch.int64), torch.tensor(special_tokens, dtype=torch.int64), torch.tensor(target_move_index, dtype=torch.int64)
            if self.masking:
                legal_moves_list = json.loads(row[4])
                yield (board_state_tensor, 
                    special_token_tensor,
                    target_move_tensor,
                    legal_moves_list) #returned as list initially to allow for padding
            else:
                yield (board_state_tensor, 
                    special_token_tensor,
                    target_move_tensor)
        
        conn.close()
    
    def flip_uci(self, uci_move_index):
        uci_move_string = self.index_to_move[uci_move_index]
        uci_move_list = list(uci_move_string)
        uci_move_list[1]=str(9 - int(uci_move_list[1]))
        uci_move_list[3]=str(9 - int(uci_move_list[3]))
        flip_uci_move_string = ''.join(uci_move_list)
        return self.move_to_index[flip_uci_move_string]

def pad_collate(batch):
    # Unpack the batch into respective tensors and lists
    board_states = torch.stack([data[0] for data in batch])       # Already tensors, just stack them
    special_tokens = torch.stack([data[1] for data in batch])      # Already tensors, just stack them
    target_moves = torch.stack([data[2] for data in batch])        # Already tensors, just stack them

    # Handle legal_moves if masking is enabled (these are lists, need padding)
    if len(batch[0]) == 4:  # Check if legal_moves are present
        legal_moves = [torch.tensor(data[3], dtype=torch.int64) for data in batch]
        
        # Pad legal_moves to the same length automatically
        legal_moves_padded = pad_sequence(legal_moves, batch_first=True, padding_value=-1) #pad_sequence is an imported function
    else:
        legal_moves_padded = None

    return board_states, special_tokens, target_moves, legal_moves_padded