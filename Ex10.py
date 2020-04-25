#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabian
"""

import numpy
import sys
from scipy.optimize import linprog
from Utils import transform_NN_to_NA
import matplotlib.pyplot as pyplot


def frange(start, stop, step):
     i = start
     while i < stop:
         yield i
         i += step

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


delta = 0.05
initial_value = 0
end_value = 1
#lambdas = frange(initial_value, end_value, delta)
lambdas = numpy.arange(initial_value, end_value, delta)

lagranges = []
for idx in range(len(lambdas)):
    distance_vector_sub_lambda = distance_vector + lambdas[idx] * maximum_time_available
    result = linprog(c=distance_vector_sub_lambda, 
                     A_eq=matrix_node_arch, 
                     b_eq=beq, 
                     bounds=bounds, 
                     method='simplex')
    #print(result)
    lagranges.append(result.fun - lambdas[idx] * maximum_time_available )
    
    
max_lagrange = max(lagranges)
idx = lagranges.index(max_lagrange)

pyplot.title('Lagrange Relaxation')
pyplot.xlabel('lambda')
pyplot.ylabel('Lagrange(lambda)')
pyplot.plot(lambdas[idx], max_lagrange, 'ro')
pyplot.text(lambdas[idx], max_lagrange, "($\lambda$,costs($\lambda$)) = (%0f, %0f)" % (lambdas[idx], max_lagrange))  
pyplot.plot(lambdas, lagranges)
pyplot.grid()
pyplot.show()







