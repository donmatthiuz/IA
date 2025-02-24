import os
from process import *
# Colores ANSI para la terminal
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

diccionario_imagenes = {
    1: "Test.bmp",
    2: "Test2.bmp",
    3: "Prueba Lab1.bmp"
}

def limpiar_pantalla():
    os.system("cls" if os.name == "nt" else "clear")

def mostrar_menu_imagenes():
    limpiar_pantalla()
    print(f"""
{CYAN}╔═══════════════════════════════════╗
║      {YELLOW}★ SELECCIONA UNA IMAGEN ★{CYAN}    ║
╠═══════════════════════════════════╣
║ {GREEN}1.{RESET} Test.bmp                       ║
║ {GREEN}2.{RESET} Test2.bmp                      ║
║ {GREEN}3.{RESET} Prueba Lab1.bmp                ║
║ {GREEN}4.{RESET} Salir                          ║
╚═══════════════════════════════════╝
""")

def mostrar_menu_algoritmos():
    limpiar_pantalla()
    print(f"""
{CYAN}╔═══════════════════════════════════╗
║      {YELLOW}★ SELECCIÓN DE ALGORITMO ★{CYAN}   ║
╠═══════════════════════════════════╣
║ {GREEN}1.{RESET} Breadth First Search (BFS)     ║
║ {GREEN}2.{RESET} Depth First Search (DFS)       ║
║ {GREEN}3.{RESET} A* (A-Star)                    ║
║ {GREEN}4.{RESET} Volver al menú de imágenes     ║
╚═══════════════════════════════════╝
""")

def mostrar_menu_euristica():
    limpiar_pantalla()
    print(f"""
{CYAN}╔═══════════════════════════════════╗
║      {YELLOW}★ ELEGIR HEURISTICA ★{CYAN}   ║
╠═══════════════════════════════════╣
║ {GREEN}1.{RESET} MANHATTAN     ║
║ {GREEN}2.{RESET} EUCLIDIANA       ║
╚═══════════════════════════════════╝
""")
def menu():
    incio1_ = True
    inicio2_ = True
    while incio1_:
        mostrar_menu_imagenes()
        opcion_imagen = input(f"{BLUE}Selecciona una imagen:{RESET} ")

        if opcion_imagen in ["1", "2", "3"]:
            imagen_name = diccionario_imagenes[int(opcion_imagen)]
            imagen = cargar_imagen(imagen_name)
            matriz, inicio, metas, filas, cols = imagen_a_matriz(imagen, tam_bloque=10)
            dibujar_matriz(matriz)
            if inicio is None or not metas:
              print("Error: La imagen no tiene un inicio o metas válidas.")
              break
            while inicio2_:
                mostrar_menu_algoritmos()
                opcion_algoritmo = input(f"{BLUE}Elige un algoritmo:{RESET} ")

                if opcion_algoritmo in ["1", "2", "3"]:
                  
                  maze = matriz
                  start_pos = inicio
                  goal_positions = metas

                  problem = MazeProblem(maze, start_pos, goal_positions)
                  search = None
                  solution_path = None
                  if opcion_algoritmo == "1":
                      search = GraphSearch(problem)
                      solution_path = search.breadth_first_search()
                  elif opcion_algoritmo == "2":
                      search = GraphSearch(problem)
                      solution_path = search.depth_first_search()
                  
                  elif opcion_algoritmo == "3":
                      mostrar_menu_euristica()
                      euristica = input(f"{BLUE}Elige una heuristica para A*:{RESET} ")
                      
                      if euristica == "1":
                          search = GraphSearch(problem,heuristica_manhattan)
                          solution_path = search.a_Star()
                      elif euristica == "2":
                          search = GraphSearch(problem,heuristica_euclidiana)
                          solution_path = search.a_Star()
                  
                  dibujar_camino(matriz=matriz, camino=solution_path)
                  print(f"\n{GREEN}Ejecutando algoritmo...{RESET}")

                elif opcion_algoritmo == "4":
                    inicio2_ = False
                else:
                    print(f"\n{RED}Opción inválida. Inténtalo de nuevo.{RESET}")
                
                input(f"\n{CYAN}Presiona Enter para continuar...{RESET}")

        elif opcion_imagen == "4":
            print(f"\n{RED}Saliendo del programa...{RESET}")
            incio1_ = False
        else:
            print(f"\n{RED}Opción inválida. Inténtalo de nuevo.{RESET}")

        input(f"\n{CYAN}Presiona Enter para continuar...{RESET}")

# Ejecutar el menú
menu()

