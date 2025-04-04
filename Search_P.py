import random
import time
from abc import ABC, abstractmethod

class Search_P(ABC):
    DAYS = ["lunes", "martes", "miércoles"]
    exams = ["A", "B", "C", "D", "E", "F", "G"]
    
    students = {
        "Est1": ["A", "B", "C"],
        "Est2": ["B", "D", "E"],
        "Est3": ["C", "E", "F"],
        "Est4": ["A", "F", "G"]
    }
    
    def __init__(self):
        pass
    
    @abstractmethod
    def search(self):
        pass
    

    def is_valid(self, assignment, exam = None, dia = None):
      for student, courses in self.students.items():
          exams_per_day = {}
          for course in courses:
              if course in assignment:
                  day = assignment[course]
                  if day in exams_per_day:
                      return False
                  exams_per_day[day] = course

      if len(set(assignment.values())) < len(assignment.values()):
          return False

      return True


    def random_assignment(self):
      return {exam: random.choice(self.DAYS) for exam in self.exams}
    

    def count_violations(self, assignment):
      violations = 0

      for student, courses in self.students.items():
          exams_per_day = {}
          for course in courses:
              if course in assignment:
                  day = assignment[course]
                  if day in exams_per_day:
                      violations += 1
                  exams_per_day[day] = course

      return violations
    
    def get_neighbors(self, assignment):
      neighbors = []
      for exam in self.exams:
          current_day = assignment[exam]
          for day in self.DAYS:
              if day != current_day:
                  new_assignment = assignment.copy()
                  new_assignment[exam] = day
                  neighbors.append(new_assignment)
      return neighbors
    

    def print_solution(self, resultado):
        print("\nVerificación final:")
        for student, courses in self.students.items():
            days_with_exams = {}
            for course in courses:
                day = resultado[course]
                if day in days_with_exams:
                    print(f"Conflicto! {student} tiene {course} y {days_with_exams[day]} el {day}")
                else:
                    days_with_exams[day] = course
            exams_scheduled = [f"{course}({resultado[course]})" for course in courses]
            print(f"{student}: {', '.join(exams_scheduled)}")

        
    
    


class LocalSearch(Search_P):
    def search(self):
        max_iterations = 1000
        start_time = time.time()
        current = self.random_assignment()
        current_violations = self.count_violations(current)

        for _ in range(max_iterations):
            neighbors = self.get_neighbors(current)
            neighbors = sorted(neighbors, key=lambda x: self.count_violations(x))

            if not neighbors:
                break

            best_neighbor = neighbors[0]
            best_violations = self.count_violations(best_neighbor)


            if best_violations >= current_violations:
                break

            current = best_neighbor
            current_violations = best_violations

            if current_violations == 0:
                break

        end_time = time.time()
        return current, current_violations, end_time - start_time
    
    def is_valid(self, assignment, exam=None, dia=None):
        for student, courses in self.students.items():
          exams_per_day = {}
          for course in courses:
              if course in assignment:
                  day = assignment[course]
                  if day in exams_per_day:
                      return False 
                  exams_per_day[day] = course
        if len(set(assignment.values())) < len(assignment.values()):
            return False 

        return True
    


class Backtraking_Search(Search_P):
    def search(self):
      start_time = time.time()
      def get_exam_constraints(exam):
          count = 0
          for student, courses in self.students.items():
              if exam in courses:
                  count += len(courses)
          return count
      
      sorted_exams = sorted(self.exams, key=get_exam_constraints, reverse=True)

      def backtrack(assignment):
          if len(assignment) == len(self.exams):
              return assignment
          unassigned = [exam for exam in sorted_exams if exam not in assignment]
          if not unassigned:
              return assignment
          exam = unassigned[0]

          for day in self.DAYS:
              if self.is_valid(assignment, exam, day):
                  assignment[exam] = day
                  result = backtrack(assignment)
                  if result is not None:
                      return result
                  del assignment[exam]
              
          
          return None
      solution = backtrack({})
      end_time = time.time()
      
      return solution, end_time - start_time
    
    
    def is_valid(self, assignment, exam=None, dia=None):
        for student, courses in self.students.items():
          if exam in courses:
              for course in courses:
                  if course in assignment and course != exam and assignment[course] == dia:
                      return False
    
        return True
    


class Beam_Search(Search_P):
    def search(self):
      k=3
      max_iterations=1000
      start_time = time.time()
      current = self.random_assignment()
      current_violations = self.count_violations(current)


      beam = [current]

      for _ in range(max_iterations):
          new_beam = []


          for assignment in beam:
              neighbors = self.get_neighbors(assignment)
              neighbors = sorted(neighbors, key=lambda x: self.count_violations(x))

          
              new_beam.extend(neighbors[:k])

          
          new_beam = sorted(new_beam, key=lambda x: self.count_violations(x))
          beam = new_beam[:k]

          
          best_assignment = beam[0]
          best_violations = self.count_violations(best_assignment)

          if best_violations == 0:
              end_time = time.time()
              return best_assignment, best_violations, end_time - start_time
          
      end_time = time.time()
      return beam[0], self.count_violations(beam[0]), end_time - start_time
