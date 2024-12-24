import re
from collections import Counter
from itertools import chain

import chess
from torch import zeros, Tensor


class FenParser():
    def __init__(self, fen_str):
        self.fen_str = fen_str

    def parse(self):
        ranks = self.fen_str.split(" ")[0].split("/")
        pieces_on_all_ranks = [self.parse_rank(rank) for rank in ranks]
        return pieces_on_all_ranks

    def parse_rank(self, rank):
        rank_re = re.compile("(\\d|[kqbnrpKQBNRP])")
        piece_tokens = rank_re.findall(rank)
        pieces = self.flatten(map(self.expand_or_noop, piece_tokens))
        return pieces

    def flatten(self, lst):
        return list(chain(*lst))

    def expand_or_noop(self, piece_str):
        piece_re = re.compile("([kqbnrpKQBNRP])")
        retval = ""
        if piece_re.match(piece_str):
            retval = piece_str
        else:
            retval = self.expand(piece_str)

        return retval

    def expand(self, num_str):
        return int(num_str)*" "

    def white_to_move(self):
        return True if self.fen_str.split()[1] == 'w' else False

    def can_castle(self, color, side):
        castling_rights = self.fen_str.split()[2]

        if castling_rights == '-':
            return False

        if color == 'w':
            if side == 'k':
                return True if 'K' in castling_rights else False
            elif side == 'q':
                return True if 'Q' in castling_rights else False

        if color == 'b':
            if side == 'k':
                return True if 'k' in castling_rights else False
            elif side == 'q':
                return True if 'q' in castling_rights else False

        return False

    def get_board(self):
        return chess.Board(self.fen_str)

    def legal_moves(self):
        return self.get_board().legal_moves

    def piece_counts(self):
        """remove piece counts e.g. {'P': 8, 'N': 2, 'B': 2, 'R': 2, 'Q': 0}"""
        position = self.fen_str.split()[0]
        position = position.translate(str.maketrans("", "", "12345678/"))
        position = position.upper()
        return Counter(position)

    def piece_to_tensor(self, piece: str) -> Tensor:
        """node features: these represent the pieces on the squares (nodes)"""
        piece = chess.Piece.from_symbol(piece)
        node_feature = zeros(13)
        node_feature[piece.__hash__() + 1] = 1
        return node_feature
