from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from Search_P import *

console = Console()

# Función para mostrar el menú
def show_menu():
    console.print(Panel("[bold cyan]Selecciona un algoritmo de búsqueda[/bold cyan]", style="bold blue"))
    console.print("[bold green]1. Local Search[/bold green]")
    console.print("[bold green]2. Backtracking[/bold green]")
    console.print("[bold green]3. Beam Search[/bold green]")
    console.print("[bold red]4. SALIR[/bold red]")


def choose_algorithm():
    manejo = True
    while manejo:
        show_menu()
        option = Prompt.ask("[bold yellow]Elige un algoritmo (1, 2, 3):[/bold yellow]")

        if option == "1":
            console.print("[bold green]Has elegido Local Search[/bold green]")
            algoritmo = LocalSearch()
            solution, violations, duration = algoritmo.search()
            print("Violaciones:", violations)
            print(f"Tiempo de ejecución: {duration:.4f} segundos")
            algoritmo.print_solution(solution)


            
            pass
        elif option == "2":
            console.print("[bold green]Has elegido Backtracking[/bold green]")
            algoritmo = Backtraking_Search()
            solution, duration = algoritmo.search()
            print(f"Tiempo de ejecución: {duration:.4f} segundos")
            algoritmo.print_solution(solution)
          
            pass
        elif option == "3":
            console.print("[bold green]Has elegido Beam Search[/bold green]")
            algoritmo = Beam_Search()
            solution, violations, duration = algoritmo.search()
            print("Violaciones:", violations)
            print(f"Tiempo de ejecución: {duration:.4f} segundos")
            algoritmo.print_solution(solution)
            pass
        
        elif option== "4":
            console.print("[bold red]Acabas de salir[/bold red]")
            manejo = False
        else:
            console.print("[bold red]Opción inválida, elige 1, 2 o 3[/bold red]")
            continue


choose_algorithm()
