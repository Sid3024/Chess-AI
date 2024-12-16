import os
import math
import time
import chess
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
    def __init__(self, db_path, n_limit, token_encoding_scheme, puzzle_complexity, index_to_move, move_to_index, masking=False, min_rating=0):
        self.db_path = db_path
        self.n_limit = n_limit  # Optional limit on the number of data points
        self.masking = masking
        self.token_encoding_scheme = token_encoding_scheme
        self.puzzle_complexity = puzzle_complexity #n moves to solve
        self.index_to_move = index_to_move
        self.move_to_index = move_to_index
        self.min_rating = min_rating

    def __iter__(self):
        return self.data_generator()

    def data_generator(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Calculate limits for each split
        
        
        if self.n_limit is not None:
            total_query = "SELECT COUNT(*) FROM chess_analysis;"
            cursor.execute(total_query)
            total_rows = cursor.fetchone()[0]
            total_rows = min(total_rows, self.n_limit)  # Adjust total_rows based on n_limit
            query = f"SELECT fen, moves FROM chess_analysis LIMIT {total_rows} WHERE num_of_moves = {self.puzzle_complexity} AND rating > {self.min_rating};"
        else:
            query = f"SELECT fen, moves FROM chess_analysis WHERE num_of_moves = {self.puzzle_complexity} AND rating > {self.min_rating};"
        


        # Prepare the query based on the split
        #query = f"SELECT fen, moves FROM chess_analysis LIMIT {total_rows} WHERE num_of_moves = {self.puzzle_complexity};"

        cursor.execute(query)
        for row in cursor.fetchall():
            previous_fen = row[0]
            previous_move = json.loads(row[1])[0]
            board = chess.Board(previous_fen)
            board.push(chess.Move.from_uci(self.index_to_move[previous_move]))
            fen = board.fen()
            board_state, special_tokens, turn_encoding = self.fen_to_vector(fen)
            target_move_index = json.loads(row[1])[1]
            if turn_encoding == 1 and self.token_encoding_scheme % 2 == 0:
                target_move_index = self.flip_uci(target_move_index)
                turn_encoding_tensor = torch.tensor(turn_encoding, dtype=torch.int64)
            else:
                turn_encoding_tensor = None
            board_state_tensor, special_token_tensor, target_move_tensor = torch.tensor(board_state, dtype=torch.int64), torch.tensor(special_tokens, dtype=torch.int64), torch.tensor(target_move_index, dtype=torch.int64)
            if self.masking:
                #TODO
                board = chess.board(fen)
                legal_moves_list = None
                yield (board_state_tensor, 
                    special_token_tensor,
                    target_move_tensor,
                    legal_moves_list) #returned as list initially to allow for padding
            else:
                yield (board_state_tensor, 
                    special_token_tensor,
                    target_move_tensor,
                    turn_encoding_tensor)
        
        conn.close()

    
    def fen_to_vector(self, fen):
        fen_parts = fen.split(" ")
        rows = fen_parts[0].split("/")
        turn = fen_parts[1]
        half_moves = fen_parts[4]
        full_moves = fen_parts[5]
        turn_encoding = 0 if turn == "w" else 1

        # Initialize the position array with the special token at the start
        position = []  # Special token
        piece_dict = {
            " ": 0, "p": 1, "n": 2, "b": 3, "r": 4, "q": 5, "k": 6,
            "P": 7, "N": 8, "B": 9, "R": 10, "Q": 11, "K": 12
        }

        index = 0
        # Loop over each row of the board, swap case and reverse if it's Black's turn
        for row in rows:
            for square in row:
                if square.isdigit():
                    # Add empty squares (represented by 1s) directly
                    position.extend([0] * int(square))
                else:
                    # Add piece codes from the piece_dict
                    position.append(piece_dict.get(square, 0))
        
        # Handle castling rights
        castling_rights = fen_parts[2]
        special_tokens = [1 if c in castling_rights else 0 for c in "KQkq"]

        # Handle en passant square
        en_passant = fen_parts[3]
        if en_passant == "-":
            special_tokens.extend([0] * 9)
        else:
            file_index = ord(en_passant[0]) - 97
            special_tokens.extend([1] + [0] * file_index + [1] + [0] * (7 - file_index))
        

        if self.token_encoding_scheme % 2 == 1: #if encoding 1 or 3, add turn token.
            special_tokens.append(turn_encoding)

        return position, special_tokens, turn_encoding#, half_moves, full_moves
    
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