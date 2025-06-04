from pyepo.model.grb import shortestPathModel
from solver import solver_partial_shortest_path
import numpy as np

h=5
l=5
grid = (h,l)
model = shortestPathModel(grid)
num_nodes = grid[0]*grid[1]
arcs = model._getArcs()
incidence_matrix = np.zeros((num_nodes, len(arcs)), dtype=int)
for i in range(num_nodes):
    for j, arc in enumerate(arcs):
        if i == arc[0]:
            incidence_matrix[i][j] = -1
        elif i == arc[1]:
            incidence_matrix[i][j] = 1
b = np.array([-1] + [0]*(num_nodes-2) + [1])

model = solver_partial_shortest_path(incidence_matrix, b)
model.setObj(np.arange(len(arcs)))
x, s = model.solve()
 

print(s)
for i, arc in enumerate(arcs):
    print(f" {arc} : {x[i]}", end ="\t\t")
    if i%4==3:    
        print()