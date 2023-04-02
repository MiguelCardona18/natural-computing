import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy import trapz
import random
import decimal

def sphere(x_vector, dim):
    total_sum = 0
    for x_value in x_vector:
        total_sum += x_value**2

    return total_sum

def rastrigin(x_vector, dim):
    total_sum = 0
    for x_value in x_vector:
        total_sum += (x_value**2)-(10*math.cos((2*math.pi*x_value)))
    total_sum += 10*dim
    return total_sum

class Particle(): # all the material that is relavant at the level of the individual particles
    
    def __init__(self, dim, minx, maxx, function):
        self.position = np.random.uniform(low=minx, high=maxx, size=dim)
        self.velocity = np.random.uniform(low=-0.1, high=0.1, size=dim)
        self.best_particle_pos = self.position
        self.dim = dim
        self.function = function
        if function == 0:
            self.fitness = sphere(self.position,self.dim)
        else:
            self.fitness = rastrigin(self.position,self.dim)
        
        self.best_particle_fitness = self.fitness   # we couldd start with very large number here, 
                                                    #but the actual value is better in case we are lucky 
                
    def setPos(self, pos):
        self.position = pos
        if self.function == 0:
            self.fitness = sphere(self.position,self.dim)
        else:
            self.fitness = rastrigin(self.position,self.dim)
        
        if self.fitness<self.best_particle_fitness:     # to update the personal best both 
                                                        # position (for velocity update) and
                                                        # fitness (the new standard) are needed
                                                        # global best is update on swarm leven
            self.best_particle_fitness = self.fitness
            self.best_particle_pos = pos

    def updateVel(self, inertia, a1, a2, best_self_pos, best_swarm_pos):
                # Here we use the canonical version
                # V <- inertia*V + a1r1 (peronal_best - current_pos) + a2r2 (global_best - current_pos)
        cur_vel = self.velocity
        r1 = np.random.uniform(low=0, high=1, size = self.dim)
        r2 = np.random.uniform(low=0, high=1, size = self.dim)
        a1r1 = np.multiply(a1, r1)
        a2r2 = np.multiply(a2, r2)
        best_self_dif = np.subtract(best_self_pos, self.position)
        best_swarm_dif = np.subtract(best_swarm_pos, self.position)
                    # the next line is the main equation, namely the velocity update, 
                    # the velocities are added to the positions at swarm level
        new_vel = inertia*cur_vel + np.multiply(a1r1, best_self_dif) + np.multiply(a2r2, best_swarm_dif)
        self.velocity = new_vel
        return new_vel

def PSO(w1, a11, a21, w2, a12, a22, dim, population_size, time_steps, search_range, function):
    swarm1 = [Particle(dim,-search_range,search_range, function) for i in range(int(population_size/2))]
    swarm2 = [Particle(dim,-search_range,search_range, function) for i in range(int(population_size/2))]
    best_swarm_pos = np.random.uniform(low=-5.12, high=5.12, size=dim)
    best_swarm_fitness = 1e100
    finish = False
    for t in range(time_steps):
        if finish:
            break
        for p in range(len(swarm1)):
            if finish:
                break
            for s in range(2):
                if s == 0:
                    particle = swarm1[p]
                    w = w1
                    a1 = a11
                    a2 = a21
                elif s == 1:
                    particle = swarm2[p]
                    w = w2
                    a1 = a12
                    a2 = a22
                    
                new_position = particle.position + particle.updateVel(w, a1, a2, particle.best_particle_pos, best_swarm_pos)
                if best_swarm_fitness < 0.0001:
                    fininsh = True
                    break       
                if new_position@new_position > 1.0e+18: # The search will be terminated if the distance 
                                                        # of any particle from center is too large
                    print('Time:', t,'Best Pos:',best_swarm_pos,'Best Fit:',best_swarm_fitness)
                    # raise SystemExit('Most likely divergent: Decrease parameter values')
                    fininsh = True
                    break

                particle.setPos(new_position)

                if function == 0:
                    new_fitness = sphere(new_position,dim)
                else:
                    new_fitness = rastrigin(new_position,dim)

                if new_fitness < best_swarm_fitness:   # to update the global best both 
                                                            # position (for velocity update) and
                                                            # fitness (the new group norm) are needed
                    best_swarm_fitness = new_fitness
                    best_swarm_pos = new_position
    return best_swarm_fitness