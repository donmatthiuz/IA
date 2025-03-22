import math
import numpy as np
import pygame
import sys
from R_estado import TDAgent
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
WINDOW_LENGTH = 4

import numpy as np
import pygame
import random

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
    
    def ver_posiciones_vacias(self, board):
        valid_locations = []
        for col in range(COLS):
            if self.is_valid_move(board,col):
                valid_locations.append(col)
        return valid_locations
    
    def obtener_siguiente_row_abierta(self,board, col):
        for r in range(ROWS):
            if board[r][col] == 0:
                return r
    
    def es_nodo_terminal(self, board):
        return self.check_winner(board) or len(self.ver_posiciones_vacias(board)) == 0

    def draw_board(self):
        self.screen.fill(BLACK)
        for c in range(COLS):
            for r in range(ROWS):
                pygame.draw.rect(self.screen, BLUE, (c * SQUARESIZE, (r + 1) * SQUARESIZE, SQUARESIZE, SQUARESIZE))
                color = BLACK if self.board[r, c] == 0 else (RED if self.board[r, c] == 1 else YELLOW)
                pygame.draw.circle(self.screen, color, (c * SQUARESIZE + SQUARESIZE // 2, (r + 1) * SQUARESIZE + SQUARESIZE // 2), RADIUS)
        pygame.display.update()

    def is_valid_move(self, board, col):
        return board[0, col] == 0

    def drop_piece(self, board, col):
        for row in range(ROWS - 1, -1, -1):
            if board[row, col] == 0:
                board[row, col] = self.current_player
                return True
        return False
    

    def evaluar_jugada(self, window, pieza_oponente):
        score = 0
        if window.count(self.current_player) == 4:
            score += 100
        elif window.count(self.current_player) == 3 and window.count(0) == 1:
            score += 5
        elif window.count(self.current_player) == 2 and window.count(0) == 2:
            score += 2

        if window.count(pieza_oponente) == 3 and window.count(0) == 1:
            score -= 4

        return score
    
    def punta_posicion(self, board, mio, oponenete):
        score = 0

        ## Score center column
        center_array = [int(i) for i in list(board[:, COLS//2])]
        center_count = center_array.count(mio)
        score += center_count * 3

        ## Score Horizontal
        for r in range(ROWS):
            row_array = [int(i) for i in list(board[r,:])]
            for c in range(COLS-3):
                window = row_array[c:c+WINDOW_LENGTH]
                score += self.evaluar_jugada(window, oponenete)

       
        for c in range(COLS):
            col_array = [int(i) for i in list(board[:,c])]
            for r in range(ROWS-3):
                window = col_array[r:r+WINDOW_LENGTH]
                score += self.evaluar_jugada(window, oponenete)

        
        for r in range(ROWS-3):
            for c in range(COLS-3):
                window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
                score += self.evaluar_jugada(window, oponenete)

        for r in range(ROWS-3):
            for c in range(COLS-3):
                window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
                score += self.evaluar_jugada(window, oponenete)

        return score

    def check_winner(self, board):
        for r in range(ROWS):
            for c in range(COLS - 3):
                if self.check_line(board[r, c:c + 4]):
                    return board[r, c]

        for r in range(ROWS - 3):
            for c in range(COLS):
                if self.check_line(board[r:r + 4, c]):
                    return board[r, c]

        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if self.check_line([board[r + i, c + i] for i in range(4)]):
                    return board[r, c]

        for r in range(3, ROWS):
            for c in range(COLS - 3):
                if self.check_line([board[r - i, c + i] for i in range(4)]):
                    return board[r, c]

        return 0

    def check_line(self, line):
        return all(x == line[0] and x != 0 for x in line)

    def switch_player(self):
        self.current_player = 3 - self.current_player

class ConnectFourGameHuman(ConnectFourGame):
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
                    if self.is_valid_move(self.board,col):
                        self.drop_piece(self.board,col)
                        self.draw_board()
                        winner = self.check_winner(self.board)
                        print(winner)
                        if winner:
                            print(f"¡Jugador {winner} gana!")
                            label = self.myfont.render(f"¡Jugador {winner} gana!", 1, WHITE)
                            self.screen.blit(label, (40,10))
                            pygame.display.update()
				            
                            pygame.time.delay(2000)
                            self.running = False
                        self.switch_player()
        pygame.quit()


        


class ConnectFourGameAI:
    def __init__(self, play_pruning=False):
        self.agent = TDAgent()
        self.board = np.zeros((6, 7), dtype=int)
        self.play_pruning = play_pruning  

    def run(self):
        running = True
        while running:
            move = self.agent.choose_action(self.board)
            prev_board = self.board.copy()
            self.agent.drop_piece(self.board, move, 1)
            
            if self.agent.check_winner(self.board) == 1:
                reward = 1
                running = False
            elif len(self.agent.get_valid_moves(self.board)) == 0:
                reward = 0
                running = False
            else:
                reward = -0.01  # Penalización

            self.agent.update_q_value(prev_board, move, reward, self.board)

            if not running:
                self.agent.save_q_table()
                break

class ConnectFourGameAI_vsAI(ConnectFourGame):
    def __init__(self):
        super().__init__()
        self.agent1 = TDAgent()  # First AI using TD-learning
        self.agent2 = MinimaxAgent(depth=5)  # Second AI using Minimax

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            pygame.time.delay(500)  # Delay for better visualization

            if self.current_player == 1:
                # AI1 (TD-Learning) chooses a move
                state = self.agent1.state_representation(self.board)
                best_move = self.agent1.choose_action(state, self.get_valid_moves())
                
                self.drop_piece(self.board, best_move)
                self.draw_board()

                if self.check_winner(self.board):
                    label = self.myfont.render(f"TD Learning AI Wins!", 1, WHITE)
                    self.screen.blit(label, (5, 5))
                    pygame.display.update()
                    pygame.time.delay(2000)
                    self.running = False
                else:
                    self.switch_player()

            elif self.current_player == 2:
                # AI2 (Minimax) chooses a move
                best_move = self.agent2.get_best_move(self.board, False)

                self.drop_piece(self.board, best_move)
                self.draw_board()

                if self.check_winner(self.board):
                    label = self.myfont.render(f"Minimax AI Wins!", 1, WHITE)
                    self.screen.blit(label, (5, 5))
                    pygame.display.update()
                    pygame.time.delay(2000)
                    self.running = False
                else:
                    self.switch_player()

        pygame.quit()
