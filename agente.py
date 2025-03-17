import math
import random

import numpy as np

def minimax(mi_juego, 
						depth, 
						alpha, 
						beta, 
						maximizingPlayer, IA, Enemigo):
	valid_locations = mi_juego.ver_posiciones_vacias()
	is_terminal = mi_juego.es_nodo_terminal()
	if depth == 0 or is_terminal:
		if is_terminal:
			ganador_es = mi_juego.check_winner()
			if ganador_es == IA:
				return (None, 100000000000000)
			elif ganador_es == Enemigo:
				return (None, -10000000000000)
			else: # empate si existe jajaj xd
				return (None, 0)
		else: # Depth is zero
			return (None, mi_juego.punta_posicion(mi_juego.board, Enemigo))
	if maximizingPlayer:
		value = -math.inf
		column = random.choice(valid_locations)
		for col in valid_locations:
			mi_juego.drop_piece(col)
			col_s, new_score = minimax(mi_juego, depth-1, alpha, beta, False, IA, Enemigo)
			if new_score > value:
				value = new_score
				column = col
			alpha = max(alpha, value)
			if alpha >= beta:
				break
		return column, value
	


  