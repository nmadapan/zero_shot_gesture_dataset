from platypus import NSGAII, Problem, Real, nondominated
import numpy as np

# C = [[-2, 1], [2, 1]]
# A = [[-1, 1, -1], [1, 1, -7]]

C = -1 * np.array([[6, 4, 5], [0, 0, 1]])
A = np.array([[1, 1, 2, -12], [1, 2, 1, -12], [2, 1, 1, -12]])

def belegundu(vars):
	# print(type(vars), type(vars[0]), vars[0])
	# x = vars[0]
	# y = vars[1]
	# print(vars, type(vars))
	return np.dot(C, vars).tolist(), np.dot(A, vars+[1]).tolist()
	# return [-2*x + y, 2*x + y], [-x + y - 1, x + y - 7]

problem = Problem(3, 2, 3)
problem.types[:] = [Real(0, 4), Real(0, 4), Real(0, 4)]
problem.constraints[:] = "<=0"
problem.function = belegundu

algorithm = NSGAII(problem)
algorithm.run(10000)

feasible_solutions = [s for s in algorithm.result if s.feasible]
nondominated_solutions = nondominated(algorithm.result)

print(nondominated_solutions)
