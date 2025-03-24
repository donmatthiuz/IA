import math
import numpy as np
import pygame
import sys
from R_estado import MinimaxAgent, TDAgent


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





ROWS = 6
COLS = 7
WINDOW_LENGTH = 4

class ConnectFourGame:
    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.current_player = 1
        self.running = True

    def ver_posiciones_vacias(self, board):
        return [col for col in range(COLS) if self.is_valid_move(board, col)]

    def obtener_siguiente_row_abierta(self, board, col):
        for r in range(ROWS):
            if board[r][col] == 0:
                return r

    def es_nodo_terminal(self, board):
        return self.check_winner(board) or not self.ver_posiciones_vacias(board)

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

    def punta_posicion(self, board, mio, oponente):
        score = 0

        center_array = [int(i) for i in list(board[:, COLS // 2])]
        score += center_array.count(mio) * 3

        for r in range(ROWS):
            row_array = [int(i) for i in list(board[r, :])]
            for c in range(COLS - 3):
                score += self.evaluar_jugada(row_array[c:c + WINDOW_LENGTH], oponente)

        for c in range(COLS):
            col_array = [int(i) for i in list(board[:, c])]
            for r in range(ROWS - 3):
                score += self.evaluar_jugada(col_array[r:r + WINDOW_LENGTH], oponente)

        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                score += self.evaluar_jugada([board[r + i][c + i] for i in range(WINDOW_LENGTH)], oponente)

        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                score += self.evaluar_jugada([board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)], oponente)

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

    
    def reset(self):
        self.board = np.zeros((ROWS, COLS), dtype=int)



class ConnectFourGameAI_train:
    def __init__(self, play_pruning=False):
        self.agent = TDAgent()
        self.board = np.zeros((6, 7), dtype=int)
        self.play_pruning = play_pruning

    def run(self, episodes=1000):
        for episode in range(episodes):
            self.board = np.zeros((6, 7), dtype=int)  # Reiniciar el tablero en cada episodio
            running = True
            total_reward = 0

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
                total_reward += reward

                if not running:
                    self.agent.save_q_table()
                    break
            if self.agent.epsilon > self.agent.min_epsilon:
                self.agent.epsilon *= self.agent.epsilon_decay

            print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Epsilon: {self.agent.epsilon}")


class ConnectFourGameHuman(ConnectFourGame):
    def run(self):
        while self.running:
            print(f"Turno del jugador {self.current_player}")
            print(self.board)
            col = int(input("Elige una columna (0-6): "))
            if 0 <= col < COLS and self.is_valid_move(self.board, col):
                self.drop_piece(self.board, col)
                winner = self.check_winner(self.board)
                if winner:
                    print(f"¡Jugador {winner} gana!")
                    self.running = False
                self.switch_player()
            else:
                print("Movimiento no válido. Inténtalo de nuevo.")


class ConnectFourGameAI(ConnectFourGame):
    def __init__(self, play_pruning=False):
        super().__init__()
        self.agent = MinimaxAgent(depth=5)
        self.play_pruning = play_pruning

    def run(self):
        while self.running:
            if self.current_player == 1:
                print(self.board)
                col = int(input("Jugador 1, elige una columna (0-6): "))
                if 0 <= col < COLS and self.is_valid_move(self.board, col):
                    self.drop_piece(self.board, col)
                    if self.check_winner(self.board):
                        print(self.board)
                        print("¡Jugador 1 gana!")
                        self.running = False
                    else:
                        self.switch_player()
                else:
                    print("Movimiento no válido. Inténtalo de nuevo.")
            elif self.current_player == 2:
                print(self.board)
                print("La IA está pensando...")
                best_move = self.agent.get_best_move(self.board, self.play_pruning)
                self.drop_piece(self.board, best_move)
                print(f"La IA juega en la columna {best_move}")
                if self.check_winner(self.board):
                    print(self.board)
                    print("¡La IA gana!")
                    self.running = False
                else:
                    self.switch_player()


class Entrenamiento():
    def __init__(self, tablero):
        self.agent = TDAgent()
        self.board = tablero


    def run(self):
        alpha_range = [0.1, 0.05, 0.01]
        gamma_range = [0.9, 0.8, 0.7]
        epsilon_range = [0.9,  0.5, 0.1]
        self.agent.fine_tune_parameters(alpha_range=alpha_range, 
                                        gamma_range=gamma_range, 
                                        epsilon_range=epsilon_range)



class ConectFourGame_TDAg_IA(ConnectFourGame):
    def __init__(self, competir_contra="MINIMAX", episodios=50):
        # Llamamos al constructor de la clase base (ConnectFourGame)
        self.victorias = 0
        self.episodios = episodios
        self.competidor = competir_contra
        self.agente2 = None
        super().__init__()
        if competir_contra == "MINIMAX":
            self.agente2 = MinimaxAgent(depth=5)
        elif competir_contra == "MINIMAX_ALPHA":
            self.agente2 = MinimaxAgent(depth=5)
        
        elif competir_contra == "TD":
            self.agente2 = TDAgent()
            self.agente2.train()

        self.agente1 =  TDAgent(alpha=0.01, gamma=0.7, epsilon=0.1)   
        self.agente1.train()

    def run(self):
        while self.running:
          if self.current_player == 1:
            move = self.agente1.choose_action(self.board)
            prev_board =self.board.copy()
            self.agente1.drop_piece(self.board, move, 1)

            
            print(f"TD juega en la columna {move}")

            
            if self.agente1.check_winner(self.board) == 1:
              self.victorias += 1
              reward = 1
              self.agente1.update_q_value(prev_board, move, reward, self.board)
              print("gana TD")
              self.running = False
            else:
              reward = 0
              self.agente1.update_q_value(prev_board, move, reward, self.board)
              self.switch_player()

          elif self.current_player == 2:


            #print(self.board)

            if self.competidor == "MINIMAX" or self.competidor == "MINIMAX_ALPHA":
                alfa_beta_activate = False
                if self.competidor == "MINIMAX":
                    alfa_beta_activate = False
                elif self.competidor == "MINIMAX_ALPHA":
                    alfa_beta_activate =True
                best_move = self.agente2.get_best_move(self.board,alfa_beta_activate)
                print(f"MINIMAX 2 juega en la columna {best_move}")
                self.drop_piece(self.board, best_move)
                if self.check_winner(self.board):
                  reward = -0.01
                  self.agente1.update_q_value(prev_board, move, reward, self.board)
                  print("MINIMAX GANA")
                  self.running = False
                else:
                    self.switch_player()
            
            else:
                move = self.agente2.choose_action(self.board)
                prev_board =self.board.copy()
                self.agente2.drop_piece(self.board, move, 2)
                print(f"TD competidor juega en la columna {move}")


                if self.agente2.check_winner(self.board) == 2:
                    reward = -0.01
                    self.agente1.update_q_value(prev_board, move, reward, self.board)
                    print("gana TD competidor")
                    self.running = False
                else:
                    reward = 0
                    self.agente1.update_q_value(prev_board, move, reward, self.board)
                    self.switch_player()
    
    def analize(self):
        for ep in range(self.episodios):
            self.run()
            print(f"Episodio: {ep}")
        
        print(f"Victorias: {self.victorias}")

                
                



        





class ConnectFourGameAI_vsAI(ConnectFourGame):
    def __init__(self):
        super().__init__()
        self.agent = MinimaxAgent(depth=5)
        self.agent2 = MinimaxAgent(depth=5)


    def run(self):
        while self.running:
          if self.current_player == 1:
            print(self.board)
            best_move = self.agent.get_best_move(self.board,True)
            print(f"La IA 1 juega en la columna {best_move}")
            self.drop_piece(self.board, best_move)
            if self.check_winner(self.board):
              print(self.board)
              print("Ia con pruning gana")
              self.running = False
            else:
              self.switch_player()
          elif self.current_player == 2:
            print(self.board)
            best_move = self.agent.get_best_move(self.board,False)
            print(f"La IA 2 juega en la columna {best_move}")
            self.drop_piece(self.board, best_move)
            if self.check_winner(self.board):
              print(self.board)
              print("Ia sin pruning gana")
              self.running = False
            else:
              self.switch_player()
