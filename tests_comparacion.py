from Search_P import LocalSearch, Beam_Search, Backtraking_Search
import matplotlib.pyplot as plt

algoritmos = {
    "Búsqueda Local": LocalSearch(),
    "Beam Search": Beam_Search(),
    "Backtracking": Backtraking_Search()
}

resultados = {}

def ejecutar_algoritmo(nombre, algoritmo):
    print(f"\n{nombre}")

    if isinstance(algoritmo, Backtraking_Search):
        solucion, duracion = algoritmo.search()
        violaciones = None
    else:
        solucion, violaciones, duracion = algoritmo.search()

    es_valida = algoritmo.is_valid(solucion)

    print(f"Solucin encontrada: {solucion}")
    if violaciones is not None:
        print(f"Violaciones: {violaciones}")
    print(f"Tiempo de ejecución: {duracion:.4f} segundos")
    print("Solución válida" if es_valida else "Solución inválida")

    resultados[nombre] = {
        "solucion": solucion,
        "violaciones": violaciones,
        "tiempo": duracion,
        "valida": es_valida
    }


for nombre, algoritmo in algoritmos.items():
    ejecutar_algoritmo(nombre, algoritmo)


plt.figure(figsize=(8, 5))
nombres = list(resultados.keys())
tiempos = [resultados[nombre]['tiempo'] for nombre in nombres]

plt.bar(nombres, tiempos, color=["skyblue", "orange", "lightgreen"])
plt.ylabel("Tiempo de ejecución (segundos)")
plt.title("Comparación de Tiempos de los Algoritmos de Búsqueda")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
