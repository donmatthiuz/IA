import random

class Agent:
    def __init__(self, player_number):
        self.player_number = player_number
        self.opponent_number = 3 - player_number  # Si el jugador es 1, el oponente es 2 y viceversa
    
    def minimax(self, game, depth, is_maximizing_player):
        
        if depth == 0 or game.es_nodo_terminal(game.board):
            return game.punta_posicion(game.board, self.player_number, self.opponent_number)
        
        if is_maximizing_player:
            
            max_eval = float('-inf')
            valid_moves = game.ver_posiciones_vacias(game.board)
            best_move = None
            for col in valid_moves:
                # Simulamos el movimiento
                row = game.obtener_siguiente_row_abierta(game.board, col)
                game.board[row, col] = self.player_number
                eval = self.minimax(game, depth - 1, False)
                game.board[row, col] = 0  # Deshacer el movimiento
                
                if eval > max_eval:
                    max_eval = eval
                    best_move = col
            return max_eval if depth > 0 else best_move
        else:
        
            min_eval = float('inf')
            valid_moves = game.ver_posiciones_vacias(game.board)
            best_move = None
            for col in valid_moves:
        
                row = game.obtener_siguiente_row_abierta(game.board, col)
                game.board[row, col] = self.opponent_number
                eval = self.minimax(game, depth - 1, True)
                game.board[row, col] = 0  # Deshacer el movimiento
                
                if eval < min_eval:
                    min_eval = eval
                    best_move = col
            return min_eval if depth > 0 else best_move

    def get_best_move(self, game, depth=4):
        return self.minimax(game, depth, True)
