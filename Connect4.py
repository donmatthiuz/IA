import numpy as np
import pygame
import sys

# Definir colores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Configuración del juego
ROWS = 6
COLS = 7
SQUARESIZE = 100
RADIUS = SQUARESIZE // 2 - 5
WIDTH = COLS * SQUARESIZE
HEIGHT = (ROWS + 1) * SQUARESIZE
SIZE = (WIDTH, HEIGHT)

class ConnectFourGame:
    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.current_player = 1
        pygame.init()
        self.myfont = pygame.font.SysFont("monospace", 75)
        self.screen = pygame.display.set_mode(SIZE)
        pygame.display.set_caption("Connect Four")
        self.draw_board()
        self.running = True

    def draw_board(self):
        self.screen.fill(BLACK)
        for c in range(COLS):
            for r in range(ROWS):
                pygame.draw.rect(self.screen, BLUE, (c * SQUARESIZE, (r + 1) * SQUARESIZE, SQUARESIZE, SQUARESIZE))
                color = BLACK if self.board[r, c] == 0 else (RED if self.board[r, c] == 1 else YELLOW)
                pygame.draw.circle(self.screen, color, (c * SQUARESIZE + SQUARESIZE // 2, (r + 1) * SQUARESIZE + SQUARESIZE // 2), RADIUS)
        pygame.display.update()

    def is_valid_move(self, col):
        return self.board[0, col] == 0

    def drop_piece(self, col):
        for row in range(ROWS - 1, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                return True
        return False

    def check_winner(self):
        for r in range(ROWS):
            for c in range(COLS - 3):
                if self.check_line(self.board[r, c:c + 4]):
                    return self.board[r, c]

        for r in range(ROWS - 3):
            for c in range(COLS):
                if self.check_line(self.board[r:r + 4, c]):
                    return self.board[r, c]

        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if self.check_line([self.board[r + i, c + i] for i in range(4)]):
                    return self.board[r, c]

        for r in range(3, ROWS):
            for c in range(COLS - 3):
                if self.check_line([self.board[r - i, c + i] for i in range(4)]):
                    return self.board[r, c]

        return 0

    def check_line(self, line):
        return all(x == line[0] and x != 0 for x in line)

    def switch_player(self):
        self.current_player = 3 - self.current_player

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEMOTION:
                    pygame.draw.rect(self.screen, BLACK, (0, 0, WIDTH, SQUARESIZE))
                    x_pos = event.pos[0]
                    pygame.draw.circle(self.screen, RED if self.current_player == 1 else YELLOW, (x_pos, SQUARESIZE // 2), RADIUS)
                    pygame.display.update()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    col = event.pos[0] // SQUARESIZE
                    if self.is_valid_move(col):
                        self.drop_piece(col)
                        self.draw_board()
                        winner = self.check_winner()
                        if winner:
                            print(f"¡Jugador {winner} gana!")
                            label = self.myfont.render(f"¡Jugador {winner} gana!", 1, WHITE)
                            self.screen.blit(label, (40,10))
                            pygame.display.update()
				            
                            pygame.time.delay(2000)
                            self.running = False
                        self.switch_player()
        pygame.quit()
        


def main():
    while True:
        
        print("""
        ╔════════════════════════╗
        ║     Connect Four       ║
        ╠════════════════════════╣
        ║ 1. Jugar contra otro   ║
        ║ 2. Salir               ║
        ╚════════════════════════╝
        """)
        choice = input("Selecciona una opción: ")
        if choice == "1":
            game = ConnectFourGame()
            game.run()
        elif choice == "2":
            print("Saliendo del juego...")
            break
        else:
            print("Opción inválida. Intenta de nuevo.")
            input("Presiona Enter para continuar...")


if __name__ == "__main__":
    main()