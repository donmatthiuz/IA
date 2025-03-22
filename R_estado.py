import numpy as np

class ConnectFourState:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int)
        self.current_player = 1  # Jugador 1 inicia

    def reset(self):
        """ Reinicia el estado del juego """
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1

    def get_state(self):
        """ Devuelve una representación del estado actual """
        return self.board.copy()

    def get_feature_vector(self):
        """ Devuelve el estado del tablero como un vector de características """
        return np.append(self.board.flatten(), self.current_player)

    def is_valid_move(self, col):
        """ Verifica si una columna tiene espacio disponible """
        return self.board[0, col] == 0

    def make_move(self, col):
        """ Realiza un movimiento en la columna dada """
        if not self.is_valid_move(col):
            raise ValueError("Movimiento inválido")

        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                self.current_player *= -1  # Alternar entre 1 y -1 para cambiar de jugador
                break

    def __str__(self):
        """ Representación en texto del tablero con números de columna """
        board_str = str(self.board) + "\n" + str(list(range(self.cols)))
        return board_str

# Ejemplo de uso
if __name__ == "__main__":
    game = ConnectFourState()
    game.make_move(3)
    game.make_move(4)
    print(game)
    print("Feature Vector:", game.get_feature_vector())
