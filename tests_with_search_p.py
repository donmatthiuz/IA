from Search_P import *

local_search = LocalSearch()
beam_search = Beam_Search()
backtracking = Backtraking_Search()

def run_algorithm(algorithm, name):
    
    if isinstance(algorithm, Backtraking_Search):  
        solution, duration = algorithm.search()
        violations = None  
    else:
        solution, violations, duration = algorithm.search()

    print(f"Solución: {solution}")
    if violations is not None:
        print(f"Violación: {violations}")
    print(f"Tiempo de ejecución: {duration:.4f} segundos")

    if algorithm.is_valid(solution):
        print(f"{name} se encontró una solucin válida\n")
    else:
        print(f"{name} se encontró una solución invalida\n")

run_algorithm(local_search, "Local Search")
run_algorithm(beam_search, "Beam Search")
run_algorithm(backtracking, "Backtracking Search")
