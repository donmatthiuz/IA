import math
import random

import numpy as np

def minimax(mi_juego,
						board, 
						depth, 
						alpha, 
						beta, 
						maximizingPlayer, IA, Enemigo):
	valid_locations = mi_juego.ver_posiciones_vacias(board)
	
	is_terminal = mi_juego.es_nodo_terminal(board)
	if depth == 0 or is_terminal:
		if is_terminal:
			ganador_es = mi_juego.check_winner(board)
			if ganador_es == IA:
				return (None, 100000000000000)
			elif ganador_es == Enemigo:
				return (None, -10000000000000)
			else: # empate si existe jajaj xd
				return (None, 0)
		else: # Depth is zero
			return (None, mi_juego.punta_posicion(board, IA,Enemigo))
	if maximizingPlayer:
		value = -math.inf
		column = random.choice(valid_locations)
		print(valid_locations)
		for col in valid_locations:			
			b_copy = mi_juego.board.copy()
			mi_juego.drop_piece(b_copy,column)
			new_score = minimax(mi_juego, b_copy,depth-1, alpha, beta, True, IA, Enemigo)[1]
			if new_score > value:
				value = new_score
				column = column
			alpha = max(alpha, value)
			if alpha >= beta:
				break
		return column, value
	


  