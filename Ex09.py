#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 
@author: fabian
"""

import numpy
import sys
from scipy.optimize import linprog
from Utils import transform_NN_to_NA, get_active_archs

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

if len(distance_vector) != len(arc_idxs) or len(time_vector[0]) != len(arc_idxs):
    print("The quantity of arches and the time or distance vector don't match")
    sys.exit()
    
# check if the arch has the correct cost    
for index, arch in enumerate(arc_idxs):
    print("The arch: {} has a distance of: {} and a time {}".format(arch, distance_vector[index], time_vector[0][index]))


print("\n\n## Results ##")
for maximum_time_available in [8, 9]:
    # Optimize
    result = linprog(distance_vector, 
                     A_eq=matrix_node_arch, b_eq=beq, 
                     A_ub=time_vector, b_ub=maximum_time_available, 
                     bounds=bounds, 
                     method="simplex")

    print("\n\nShortest path with {} hr. of time available\n".format(maximum_time_available))
    print("\tThe raw solution is: {}".format(result))
    
    print("The shortest path is:")
    active_archs = get_active_archs(arc_idxs, result.x)
    for arch in active_archs:
        print("Arch {}".format(arch))
        
    print("\tThe minimum distance is {}".format(result.fun))
    print("\tThe best time is {}".format(numpy.inner(result.x, time_vector)[0]))

"""
 Conclusion

if being analyzed the results of both executions, one with maximum time of 9 hours an the other one with a maximum time of 8 hours, 
we can conclude: 

There is a feasible solution with maximum time of 9 hs, and the shortest path is [(s, 3), (3, 5), (5, t)]

The is not a feasible solution with maximum time of 8 hrs, since the solver choose part of the arches and that doesn't make sense.

For this model there is no feasible solution with a maximum time of 8 hr. 
However, we can try with other methods that relax one or more restrictions and find a solution close to the optimum solution. 
One posibility could be use relaxation lagrangian.

"""

