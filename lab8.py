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


def is_valid_2(assignment, exam, day):
    # Verificar que no haya más de un examen por día por estudiante
    for student, courses in students.items():
        if exam in courses:
            # Comprobar si el estudiante ya tiene un examen ese día
            for course in courses:
                if course in assignment and course != exam and assignment[course] == day:
                    return False
    
    return True


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





def beam_search(k=3, max_iterations=1000):
    start_time = time.time()
    current = random_assignment()
    current_violations = count_violations(current)

    # Mantener las k mejores asignaciones
    beam = [current]

    for _ in range(max_iterations):
        new_beam = []

        # Generar vecinos para cada asignación en el haz
        for assignment in beam:
            neighbors = get_neighbors(assignment)
            neighbors = sorted(neighbors, key=lambda x: count_violations(x))

            # Añadir los k mejores vecinos al nuevo haz
            new_beam.extend(neighbors[:k])

        # Ordenar el nuevo haz y mantener solo los k mejores
        new_beam = sorted(new_beam, key=lambda x: count_violations(x))
        beam = new_beam[:k]

        # Comprobar si la mejor solución es válida
        best_assignment = beam[0]
        best_violations = count_violations(best_assignment)

        if best_violations == 0:
            end_time = time.time()
            return best_assignment, best_violations, end_time - start_time

    # No encontró una solución válida
    end_time = time.time()
    return beam[0], count_violations(beam[0]), end_time - start_time






# Ejecutar Beam Search
solution, violations, duration = beam_search(k=3)

print("Solución encontrada:", solution)
print("Violaciones:", violations)
print(f"Tiempo de ejecución: {duration:.4f} segundos")




if solution:
    # Verificar que no hay estudiantes con múltiples exámenes en el mismo día
    print("\nVerificación final:")
    for student, courses in students.items():
        days_with_exams = {}
        for course in courses:
            day = solution[course]
            if day in days_with_exams:
                print(f"Conflicto! {student} tiene {course} y {days_with_exams[day]} el {day}")
            else:
                days_with_exams[day] = course
        exams_scheduled = [f"{course}({solution[course]})" for course in courses]
        print(f"{student}: {', '.join(exams_scheduled)}")








def backtracking_search():
    start_time = time.time()
    def get_exam_constraints(exam):
        count = 0
        for student, courses in students.items():
            if exam in courses:
                count += len(courses)
        return count
    
    sorted_exams = sorted(exams, key=get_exam_constraints, reverse=True)

    def backtrack(assignment):
        if len(assignment) == len(exams):
            return assignment
        unassigned = [exam for exam in sorted_exams if exam not in assignment]
        if not unassigned:
            return assignment
        exam = unassigned[0]

        for day in DAYS:
            if is_valid_2(assignment, exam, day):
                assignment[exam] = day
                result = backtrack(assignment)
                if result is not None:
                    return result
                del assignment[exam]
            
        
        return None
    solution = backtrack({})
    end_time = time.time()
    
    return solution, end_time - start_time



# Ejecutar el algoritmo
# solution, violations, duration = local_search()

# print("Solución encontrada:", solution)
# print("Violaciones:", violations)
# print(f"Tiempo de ejecución: {duration:.4f} segundos")



#ejecutar Backtraking search


solution, duration = backtracking_search()

print("Solución encontrada:", solution)
if solution:
    print("Es válida:", all(is_valid_2(solution, exam, solution[exam]) for exam in exams))
else:
    print("No se encontró solución")
print(f"Tiempo de ejecución: {duration:.4f} segundos")

# Verificar que la solución cumple con todas las restricciones
if solution:
    # Verificar que no hay estudiantes con múltiples exámenes en el mismo día
    print("\nVerificación final:")
    for student, courses in students.items():
        days_with_exams = {}
        for course in courses:
            day = solution[course]
            if day in days_with_exams:
                print(f"Conflicto! {student} tiene {course} y {days_with_exams[day]} el {day}")
            else:
                days_with_exams[day] = course
        exams_scheduled = [f"{course}({solution[course]})" for course in courses]
        print(f"{student}: {', '.join(exams_scheduled)}")