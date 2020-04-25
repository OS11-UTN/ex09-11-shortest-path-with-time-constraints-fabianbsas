#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabian
"""

import numpy
import sys
from scipy.optimize import linprog
from Utils import transform_NN_to_NA, get_active_archs
import matplotlib.pyplot as pyplot


# This matrix represent a Nodo-Nodo graph
matrix_node_node = numpy.array([[0, 1, 1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 1],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0]])


# Transform Node-Node matrix to Node-Archs
# Aeq matrix Node-Archs
# arc_idxs list of archs
matrix_node_arch, arc_idxs = transform_NN_to_NA(matrix_node_node)
beq = numpy.array([1, 0, 0, 0, 0, -1])

# Distances vector 
distance_vector = numpy.array([2, 1, 2, 5, 2, 1, 2])

# Times vector 
# It is a matrix of 1 row with the times associated with each arch
time_vector = numpy.array([[3, 1, 3, 1, 3, 3, 5]]) 

bounds = tuple([(0, None) for arcs in range(0, matrix_node_arch.shape[1])])

# Time restriction to find a shortest path
maximum_time_available = 8

if len(distance_vector) != len(arc_idxs) or len(time_vector[0]) != len(arc_idxs):
    print("The quantity of arches and the time or distance vector don't match")
    sys.exit()
    
# check if the arch has the correct cost    
for index, arch in enumerate(arc_idxs):
    print("The arch: {} has a distance of: {} and a time {}".format(arch, distance_vector[index], time_vector[0][index]))



_lambda = 0
tolerance = 10**-3
iteration = 1
diff = 9999

iteration = 1 
lambdas =[]
gradients =[]
functions = []
iterations = []
while diff > tolerance:
    distance_vector_sub_lambda = distance_vector + _lambda * time_vector
    result = linprog(c=distance_vector_sub_lambda, 
                     A_eq=matrix_node_arch, 
                     b_eq=beq, 
                     bounds=bounds, 
                     method='revised simplex')
    lagrange = result.fun - _lambda * maximum_time_available
    #gradient = (time_vector @ result.x) - maximum_time_available
    gradient = numpy.dot(time_vector[0],result.x) - maximum_time_available
    step = 1 / iteration
    aux = _lambda
    _lambda = _lambda + step * gradient
    diff = abs(_lambda - aux)    
    
    lambdas.append(_lambda)
    gradients.append(gradient)
    functions.append(result.fun)
    iterations.append(iteration)

    iteration += 1
    


print("\n\n## Results ## \n\n")
selected_archs = get_active_archs(arc_idxs,result.x )
print("\tThe archs that make the shortest path are: {} \n".format(selected_archs))
print("\tThe minimun cost is {:.2f} \n".format(result.fun))


pyplot.title('Lambda convergence')
pyplot.xlabel('Iteration')
pyplot.ylabel('lambda')
pyplot.grid()
pyplot.plot ( iterations, lambdas)
pyplot.show()

pyplot.title('Objetive function convergence')
pyplot.xlabel('lambda')
pyplot.ylabel('Objetive function')
pyplot.grid()
pyplot.plot ( lambdas, functions)
pyplot.show()



    