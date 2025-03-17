import numpy as np

class MinimaxAgent:
    def __init__(self, depth=4):
        self.depth = depth

    def minimax(self, board, depth, maximizing_player):
        valid_locations = self.get_valid_moves(board)
        is_terminal = self.is_terminal_node(board)
        
        if depth == 0 or is_terminal:
            if is_terminal:
                if self.check_winner(board) == 1:
                    return (None, 1000000)
                elif self.check_winner(board) == 2:
                    return (None, -1000000)
                else: 
                    return (None, 0)
            else:
                return (None, self.evaluate_board(board))
        
        if maximizing_player:
            value = -np.inf
            column = np.random.choice(valid_locations)
            for col in valid_locations:
                temp_board = board.copy()
                self.drop_piece(temp_board, col, 1)
                new_score = self.minimax(temp_board, depth - 1, False)[1]
                if new_score > value:
                    value = new_score
                    column = col
            return column, value
        else:
            value = np.inf
            column = np.random.choice(valid_locations)
            for col in valid_locations:
                temp_board = board.copy()
                self.drop_piece(temp_board, col, 2)
                new_score = self.minimax(temp_board, depth - 1, True)[1]
                if new_score < value:
                    value = new_score
                    column = col
            return column, value
    
    def minimax_alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        valid_locations = self.get_valid_moves(board)
        is_terminal = self.is_terminal_node(board)
        
        if depth == 0 or is_terminal:
            if is_terminal:
                if self.check_winner(board) == 1:
                    return (None, 1000000)
                elif self.check_winner(board) == 2:
                    return (None, -1000000)
                else: 
                    return (None, 0)
            else:
                return (None, self.evaluate_board(board))
        
        if maximizing_player:
            value = -np.inf
            column = np.random.choice(valid_locations)
            for col in valid_locations:
                temp_board = board.copy()
                self.drop_piece(temp_board, col, 1)
                new_score = self.minimax_alpha_beta(temp_board, depth - 1, alpha, beta, False)[1]
                if new_score > value:
                    value = new_score
                    column = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return column, value
        else:
            value = np.inf
            column = np.random.choice(valid_locations)
            for col in valid_locations:
                temp_board = board.copy()
                self.drop_piece(temp_board, col, 2)
                new_score = self.minimax_alpha_beta(temp_board, depth - 1, alpha, beta, True)[1]
                if new_score < value:
                    value = new_score
                    column = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return column, value
    
    def get_valid_moves(self, board):
        return [col for col in range(board.shape[1]) if board[0, col] == 0]
    
    def is_terminal_node(self, board):
        return self.check_winner(board) != 0 or len(self.get_valid_moves(board)) == 0
    
    def drop_piece(self, board, col, player):
        for row in range(board.shape[0] - 1, -1, -1):
            if board[row, col] == 0:
                board[row, col] = player
                break
    
    def check_winner(self, board):
        for r in range(board.shape[0]):
            for c in range(board.shape[1] - 3):
                if self.check_line(board[r, c:c + 4]):
                    return board[r, c]

        for r in range(board.shape[0] - 3):
            for c in range(board.shape[1]):
                if self.check_line(board[r:r + 4, c]):
                    return board[r, c]

        for r in range(board.shape[0] - 3):
            for c in range(board.shape[1] - 3):
                if self.check_line([board[r + i, c + i] for i in range(4)]):
                    return board[r, c]

        for r in range(3, board.shape[0]):
            for c in range(board.shape[1] - 3):
                if self.check_line([board[r - i, c + i] for i in range(4)]):
                    return board[r, c]

        return 0
    
    def check_line(self, line):
        return all(x == line[0] and x != 0 for x in line)
    
    def evaluate_board(self, board):
        return np.random.randint(-10, 10)  # Función de evaluación simple (puedes mejorarla)
    
    def get_best_move(self, board, use_alpha_beta=False):
        if use_alpha_beta:
            column, _ = self.minimax_alpha_beta(board, self.depth, -np.inf, np.inf, True)
        else:
            column, _ = self.minimax(board, self.depth, True)
        return column
