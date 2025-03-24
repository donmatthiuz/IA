import numpy as np
import random
import pickle
import os
import uuid

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


class TDAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1,   epsilon_decay=0.995, min_epsilon=0.01):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.agent_id = str(uuid.uuid4())
        self.q_table = self.load_q_table()
    
    def get_q_table_filename(self):
        return f"q_table_{self.agent_id}.pkl"
    
    def load_q_table(self):
        q_table_filename = self.get_q_table_filename()
        if os.path.exists(q_table_filename):
            with open(q_table_filename, "rb") as f:
                return pickle.load(f)
        return {}

    def save_q_table(self):
        with open("q_table.pkl", "wb") as f:
            pickle.dump(self.q_table, f)

    def get_state_key(self, board):
        """Convert board state to a hashable key for Q-table."""
        return tuple(map(tuple, board))

    def get_valid_moves(self, board):
        """Return valid columns where a move can be made."""
        return [col for col in range(board.shape[1]) if board[0, col] == 0]

    def choose_action(self, board):
        """Choose an action using an epsilon-greedy policy."""
        state = self.get_state_key(board)
        valid_moves = self.get_valid_moves(board)

        if not valid_moves:  # No valid moves, return -1 to indicate a problem
            return -1

        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_moves)  # Explore
        else:
            q_values = {move: self.q_table.get((state, move), 0) for move in valid_moves}

            if not q_values:  # If q_values is empty, choose a random move
                return np.random.choice(valid_moves)

            return max(q_values, key=q_values.get)  # Exploit

    def update_q_value(self, board, action, reward, next_board):
        """Update Q-table using the Q-learning formula."""
        state = self.get_state_key(board)
        next_state = self.get_state_key(next_board)
        valid_moves = self.get_valid_moves(next_board)

        old_value = self.q_table.get((state, action), 0)
        next_max = max([self.q_table.get((next_state, move), 0) for move in valid_moves], default=0)

        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[(state, action)] = new_value

    def drop_piece(self, board, col, player):
        """Drop a piece in the given column."""
        for row in range(board.shape[0] - 1, -1, -1):
            if board[row, col] == 0:
                board[row, col] = player
                break

    def check_winner(self, board):
        """Check if there is a winner (1 for AI, -1 for opponent, 0 for no winner)."""
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
        """Check if all elements in a line are the same (and not empty)."""
        return all(x == line[0] and x != 0 for x in line)


    def train(self,  episodes=500):

        nemesys = TDAgent()
        """Train the agent using Q-learning."""
        for episode in range(episodes):
            tablero = np.zeros((6, 7), dtype=int)
         
            total_reward = 0
            running = True
            reward = 0

            while running:
                #print(tablero)
                move = self.choose_action(tablero)
                prev_board =tablero.copy()
                self.drop_piece(tablero, move, 1)


                move_enemi = nemesys.choose_action(tablero)
                nemesys.drop_piece(tablero, move_enemi, 2)


                if self.check_winner(tablero) == 1:
                    reward = 1
                    running = False
                elif len(self.get_valid_moves(tablero)) == 0:
                    reward = 0
                    running = False
                else:
                    reward = -0.01  # Penalización por no ganar por mula
                
                self.update_q_value(prev_board, move, reward, tablero)

                if not running:
                    self.save_q_table()
                    break
                                
                total_reward += reward
            
           
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay

            if (episode + 1) % 100 == 0:
               print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon}")

            
            if (episode + 1) % 100 == 0:
                self.save_q_table()
    
    def fine_tune_parameters(self, alpha_range, gamma_range, epsilon_range, episodes=500):
        best_params = None
        best_reward = -float("inf")

        for alpha in alpha_range:
            for gamma in gamma_range:
                for epsilon in epsilon_range:
                    print(f"Fine-tuning with alpha={alpha}, gamma={gamma}, epsilon={epsilon}")
                    self.alpha = alpha
                    self.gamma = gamma
                    self.epsilon = epsilon
                    self.train(episodes)
                    # Evaluate performance and choose the best parameters
                    if self.q_table:
                        total_reward = sum(self.q_table.values())
                        if total_reward > best_reward:
                            best_reward = total_reward
                            best_params = (alpha, gamma, epsilon)

        print(f"Best parameters: Alpha={best_params[0]}, Gamma={best_params[1]}, Epsilon={best_params[2]}")