# import functions
from numpy.random import rand
import random
import numpy as np
from numpy.random import choice
from numpy import asarray
from numpy import clip
from numpy import argmin
from numpy import min
from numpy import around
from scipy.integrate import simpson
from numpy import trapz
import math
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import decimal

# sphere function
def sphere(x_vector, dim):
    total_sum = 0
    for x_value in x_vector:
        total_sum += x_value**2
    return total_sum

# rastrigin function
def rastrigin(x_vector, dim):
    total_sum = 0
    for x_value in x_vector:
        total_sum += (x_value**2)-(10*math.cos((2*math.pi*x_value)))
    total_sum += 10*dim
    return total_sum

class Particle(): 
    
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
        
        self.best_particle_fitness = self.fitness
                
    def setPos(self, pos):
        self.position = pos
        if self.function == 0:
            self.fitness = sphere(self.position,self.dim)
        else:
            self.fitness = rastrigin(self.position,self.dim)
        
        if self.fitness<self.best_particle_fitness:
            self.best_particle_fitness = self.fitness
            self.best_particle_pos = pos

    def updateVel(self, inertia, a1, a2, best_self_pos, best_swarm_pos):
        cur_vel = self.velocity
        r1 = np.random.uniform(low=0, high=1, size = self.dim)
        r2 = np.random.uniform(low=0, high=1, size = self.dim)
        a1r1 = np.multiply(a1, r1)
        a2r2 = np.multiply(a2, r2)
        best_self_dif = np.subtract(best_self_pos, self.position)
        best_swarm_dif = np.subtract(best_swarm_pos, self.position)
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
                if new_position@new_position > 1.0e+18:
                    print('Time:', t,'Best Pos:',best_swarm_pos,'Best Fit:',best_swarm_fitness)
                    # raise SystemExit('Most likely divergent: Decrease parameter values')
                    fininsh = True
                    break

                particle.setPos(new_position)

                if function == 0:
                    new_fitness = sphere(new_position,dim)
                else:
                    new_fitness = rastrigin(new_position,dim)

                if new_fitness < best_swarm_fitness:
                    best_swarm_fitness = new_fitness
                    best_swarm_pos = new_position
    return best_swarm_fitness
 
# objective function
def obj(dim, x):
    total_sum = 0
    for x_value in x:
        total_sum += (x_value**2)-(10*math.cos((2*math.pi*x_value)))
    total_sum += 10*dim 
    return total_sum
 
# mutation operation
def mutation(x, F, dim):
    new_v = []
    for i in range(dim):
        a = x[0][i]
        b = x[1][i]
        c = x[2][i]
        new_v.append(a + (F * (b - c)))
    return new_v

# crossover operation
def crossover(mutated, target, dim, cr):
    # random value for each dimension
    p = rand(dim)
    trial=[]
    for i in range(dim):
        if p[i] <= cr:
            trial.append(mutated[i])
        else:
            trial.append(target[i])
    return trial
 
 
# checking boundaries
def check_bounds(mutated, bounds):
    mutated_bound=[]
    for i in range(len(mutated)):
        mutated_bound.append(clip(mutated[i], bounds[0], bounds[1]))
    return mutated_bound

 
 
def DE(pop_size, bounds, iter, F, cr, dim):
    # initialise first generation of individuals
    pop = []
    for i in range(pop_size):
        ind = []
        for d in range(dim):
            ind.append(random.uniform(bounds[0], bounds[1]))
        pop.append(ind)
    
    # evaluate population
    obj_all = [obj(dim,ind) for ind in pop]
    best_obj = min(obj_all)
    prev_obj = best_obj

    # main loop
    for i in range(iter):
        for j in range(pop_size):
            candidates=[]
            for c in range(pop_size):
                if c != j:
                    candidates.append(c)
            a,b,c = choice(candidates, 3, replace=False)
            a,b,c = pop[a],pop[b],pop[c]

            # mutation
            mutated = mutation([a, b, c], F, dim)
            # checking for bounds
            mutated = check_bounds(mutated, bounds)
            # crossover
            trial = crossover(mutated, pop[j], dim, cr)

            # evaluate using objective function (rastrigin)
            obj_target = obj(dim,pop[j])
            obj_trial = obj(dim,trial)

            # selection
            if obj_trial < obj_target:
                pop[j] = trial
                obj_all[j] = obj_trial

        # best individual of that generation
        best_obj = min(obj_all)
        if best_obj < prev_obj:
            best_vector = pop[argmin(obj_all)]
            prev_obj = best_obj
    return best_obj
 
 
# defining parameters of DE
pop_size = 30 #size of population
dim = 3 #dimensions
bounds = [-5.12, 5.12] #bounds
iter = 1001 #iterations
F = 1.5 #mutation scale factor
cr = 0.7 #crossever rate

def plot_graph():
    x = []
    y = []
    for i in np.array(np.linspace(0.001,1.999,20)):
        x.append(i)
        y.append(DE(pop_size, bounds, iter, i, cr, dim))

    plt.xlabel('Mutation Scale factor (F)')
    plt.ylabel('Best Rastrigin Function')
    plt.title('Best Rastrigin Function on different Mutation Scale factors (F)')
    plt.plot(x, y)
    plt.show()

def plot_extra_var():
    x = []
    y = []
    y2 = []
    y3 = []
    for i in range(1,21):
        print(i)
        x.append(i)
        y.append(DE(pop_size, bounds, iter, 0.5, cr, i))
        y2.append(PSO(w1=0.85, a11=0.25, a21=0.25,w2=0.85, a12=0.25, a22=0.25, dim=i,population_size=30, time_steps=1001, search_range=5.12, function=1))
        y3.append(PSO(w1=0.39, a11=0.86, a21=0.15,w2=0.28, a12=1.33, a22=2.42, dim=i,population_size=30, time_steps=1001, search_range=5.12, function=1))

    plt.xlabel('Number of Dimensions')
    plt.ylabel('Best Rastrigin Function')
    label1 = "DE: F=0.5"
    label2 = "PSO: α=0.5, ω=0.85"
    label3 = "Heterogeneous PSO: ω1=0.39, α11=0.86, α21=0.15, ω2=0.28, α12=1.33, α22=2.42"
    plt.plot(x, y, label = label1)
    plt.plot(x, y2, label = label2)
    plt.plot(x, y3, label = label3)
    plt.title('Best Swarm Performance on different number\n of Search space dimensions using different algorithms')
    plt.legend()
    plt.xticks(np.arange(min(x), max(x)+1, 1))
    plt.show()

plot_extra_var()