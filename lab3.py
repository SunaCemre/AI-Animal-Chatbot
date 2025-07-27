# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:49:24 2024

@author: sunad
"""

#!/usr/bin/env python

""" Traveling salesman problem solved using Simulated Annealing.
    Adapted from http://www.physics.rutgers.edu/~haule/681/src_MC/python_codes/salesman.py
"""

from scipy import *
from pylab import *
import random

def Distance(R1, R2):
    return sqrt((R1[0]-R2[0])**2+(R1[1]-R2[1])**2)

def TotalDistance(city, R):
    dist=0
    for i in range(len(city)-1):
        dist += Distance(R[city[i]],R[city[i+1]])
    dist += Distance(R[city[-1]],R[city[0]])
    return dist
    
def reverse(city, n):
    nn = (1+ ((n[1]-n[0]) % nct))/2 # half the lenght of the segment to be reversed
    # the segment is reversed in the following way n[0]<->n[1], n[0]+1<->n[1]-1, n[0]+2<->n[1]-2,...
    # Start at the ends of the segment and swap pairs of cities, moving towards the center.
    for j in range(int(nn)):
        k = (n[0]+j) % nct
        l = (n[1]-j) % nct
        (city[k],city[l]) = (city[l],city[k])  # swap
    
def Plot(city, R, dist):
    Pt = [R[city[i]] for i in range(len(city))]
    Pt += [R[city[0]]]
    Pt = array(Pt)
    title('Total distance='+str(dist))
    plot(Pt[:,0], Pt[:,1], '-o')
    show()

if __name__=='__main__':

    nct = 50   # Number of cities to visit
    maxTsteps = 100    # Temperature is lowered not more than maxTsteps
    Tstart = 1.0       # Starting temperature - has to be high enough
    fCool = 0.95      # Factor to multiply temperature at each cooling step
    maxSteps = 100*nct     # Number of steps at constant temperature
    maxAccepted = 10*nct   # Number of accepted steps at constant temperature
    random.seed(10)     # Remove this to have a different problem every run
    
    # Choosing city coordinates
    R=[]  # coordinates of cities are choosen randomly
    for i in range(nct):
        R.append( [random.random(),random.random()] )
    R = array(R)

    # The index table -- the order the cities are visited.
    city = list(range(nct))
    # Distance of the travel at the beginning
    dist = TotalDistance(city, R)

    # Stores points of a move
    n = zeros(6, dtype=int)
    
    T = Tstart # temperature

    Plot(city, R, dist)
    
    for t in range(maxTsteps): #For each time step;

        accepted = 0
        for i in range(maxSteps): # At each temperature;
            
            while True: # Will find two random cities n[0] and n[1] sufficiently close
                n[0] = int((nct)*random.random())     # select one city
                n[1] = int((nct-1)*random.random())   # select another city, but not the same
                if (n[1] >= n[0]): n[1] += 1   
                if (n[1] < n[0]): (n[0],n[1]) = (n[1],n[0]) # swap, because it must be: n[0]<n[1]
                nn = (n[0]+nct -n[1]-1) % nct  # number of cities not on the segment n[0]..n[1]
                if nn>=3: break
        
            # We want to have one index before and one after the two cities
            # The order hence is [n2,n0,n1,n3]
            n[2] = (n[0]-1) % nct  # index before n0  -- see figure in the lab sheet
            n[3] = (n[1]+1) % nct  # index after n2   -- see figure in the lab sheet
            
            # What would be the cost to reverse the path between city[n[0]]-city[n[1]]?
            de = Distance(R[city[n[2]]],R[city[n[1]]]) + Distance(R[city[n[3]]],R[city[n[0]]]) - Distance(R[city[n[2]]],R[city[n[0]]]) - Distance(R[city[n[3]]],R[city[n[1]]])
            
            if de<0: # the delta-E is either negative (better path found) or positive but allowed with a probability determined by T
            # -->> in the above, remove the second part of the OR to change to a Hill Climbing search (why?)
                accepted += 1
                dist += de
                # Here we reverse a segment
                reverse(city, n)
                    
            if accepted > maxAccepted: break

        # Plot
        Plot(city, R, dist)
            
        print ("T=%10.5f , distance= %10.5f , accepted steps= %d" %(T, dist, accepted))
        T *= fCool             # The system is cooled down
        if accepted == 0: break  # If the path does not want to change any more, we can stop

        
    Plot(city, R, dist)
    
    
