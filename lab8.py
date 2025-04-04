import random
import time

# Posibles días para los exámenes
DAYS = ["lunes", "martes", "miércoles"]

# Exámenes a programar (7 cursos)
exams = ["A", "B", "C", "D", "E", "F", "G"]

# Estudiantes y los cursos que toman
students = {
    "Est1": ["A", "B", "C"],
    "Est2": ["B", "D", "E"],
    "Est3": ["C", "E", "F"],
    "Est4": ["A", "F", "G"]
}

# Dominio: todos los exámenes pueden ir en cualquier día
domains = {exam: DAYS[:] for exam in exams}

# Función para verificar si una asignación es válida
def is_valid(assignment):
    # Revisar que no haya más de un examen por día por estudiante
    for student, courses in students.items():
        exams_per_day = {}
        for course in courses:
            if course in assignment:
                day = assignment[course]
                if day in exams_per_day:
                    return False  # El estudiante tiene más de un examen en el mismo día
                exams_per_day[day] = course

    # Revisar que todos los exámenes estén en días distintos
    if len(set(assignment.values())) < len(assignment.values()):
        return False  # Hay exámenes repetidos en el mismo día

    return True

# Generar una solución aleatoria (asignación de días a exámenes)
def random_assignment():
    return {exam: random.choice(DAYS) for exam in exams}

# Evaluar una asignación según la cantidad de restricciones violadas
def count_violations(assignment):
    violations = 0

    # Verificar exámenes en el mismo día para un estudiante
    for student, courses in students.items():
        exams_per_day = {}
        for course in courses:
            if course in assignment:
                day = assignment[course]
                if day in exams_per_day:
                    violations += 1  # más de un examen ese día para el estudiante
                exams_per_day[day] = course

    return violations

# Generar vecinos cambiando el día de un examen al azar
def get_neighbors(assignment):
    neighbors = []
    for exam in exams:
        current_day = assignment[exam]
        for day in DAYS:
            if day != current_day:
                new_assignment = assignment.copy()
                new_assignment[exam] = day
                neighbors.append(new_assignment)
    return neighbors

# Algoritmo de búsqueda local
def local_search(max_iterations=1000):
    start_time = time.time()
    current = random_assignment()
    current_violations = count_violations(current)

    for _ in range(max_iterations):
        neighbors = get_neighbors(current)
        neighbors = sorted(neighbors, key=lambda x: count_violations(x))

        if not neighbors:
            break

        best_neighbor = neighbors[0]
        best_violations = count_violations(best_neighbor)

        # Si no hay mejora, terminar
        if best_violations >= current_violations:
            break

        current = best_neighbor
        current_violations = best_violations

        if current_violations == 0:
            break

    end_time = time.time()
    return current, current_violations, end_time - start_time

# Ejecutar el algoritmo
solution, violations, duration = local_search()

print("Solución encontrada:", solution)
print("Violaciones:", violations)
print(f"Tiempo de ejecución: {duration:.4f} segundos")