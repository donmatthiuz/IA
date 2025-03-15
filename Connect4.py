import numpy as np

class ConnectFourBoard:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int)  # 0: vacío, 1: jugador 1, 2: jugador 2

    def print_board(self):
        """Muestra el tablero con formato más claro."""
        print("\nTablero:")
        for row in range(self.rows):
            print("| " + " ".join(str(self.board[row, col]) for col in range(self.cols)) + " |")
        print("  " + " ".join(str(i) for i in range(self.cols)))

class ConnectFour(ConnectFourBoard):
    def __init__(self, rows=6, cols=7):
        super().__init__(rows, cols)
        self.current_player = 1

    def is_valid_move(self, col):
        """Verifica si una columna tiene espacio para un movimiento."""
        return self.board[0, col] == 0

    def drop_piece(self, col):
        """Coloca una ficha en la columna especificada si es un movimiento válido."""
        if not self.is_valid_move(col):
            return False
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                self.switch_player()
                return True
        return False

    def switch_player(self):
        """Cambia al siguiente jugador."""
        self.current_player = 3 - self.current_player

    def check_winner(self):
        """Verifica si hay un ganador en filas, columnas o diagonales."""
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if self.check_line(self.board[r, c:c+4]):
                    return self.board[r, c]

        for r in range(self.rows - 3):
            for c in range(self.cols):
                if self.check_line(self.board[r:r+4, c]):
                    return self.board[r, c]

        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if self.check_line([self.board[r+i, c+i] for i in range(4)]):
                    return self.board[r, c]

        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if self.check_line([self.board[r-i, c+i] for i in range(4)]):
                    return self.board[r, c]

        return 0

    def check_line(self, line):
        """Verifica si una línea de 4 casillas contiene el mismo número (1 o 2)."""
        return all(x == line[0] and x != 0 for x in line)

game = ConnectFour()
game.print_board()

game.drop_piece(3)
game.print_board()

game.drop_piece(3)
game.print_board()

game.drop_piece(4)
game.print_board()

game.drop_piece(2)
game.print_board()