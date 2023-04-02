import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import numpy as np
from scipy.integrate import simpson
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


# print(PSO(dim=3, w=0.7, a1=2.02, a2=2.02, population_size=30, time_steps=1001, search_range=5.12))

# 3d graph showing performance with different values of alpha and omega
def plot_graph():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array(np.linspace(0.01,5.00,20))
    y = np.array(np.linspace(0.01,1.00,20))
    X, Y = np.meshgrid(x, y)
    zs = np.array([PSO(w1=y, a11=float(x/2), a21=float(x/2), w2=y, a12=float(x/2), a22=float(x/2), dim=3,population_size=30, time_steps=1001, search_range=5.12, function=0) for x,y in zip(np.ravel(X), np.ravel(Y))])
    total=0.0
    count=0
    for f in zs:
        total += f
        count +=1
    mean = total/count
    print('Mean swarm fitness: ' + str(mean))
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)

    ax.set_xlabel('α - Alpha')
    ax.set_ylabel('ω - Omega')
    ax.set_zlabel('Best Swarm Performance')

    plt.show()

# graph showing performance with different dimensions
def plot_extra_var():
    x = []
    y = []
    x2 = []
    y2 = []
    w1 = 0.85
    a11 = 0.75
    a21 = 0.75
    for i in range(3,100):
        x.append(i)
        y.append(PSO(w1=w1, a11=a11, a21=a21,w2=w1, a12=a11, a22=a21, dim=i,population_size=30, time_steps=1001, search_range=5.12, function=1))
    for i in range(3,100):
        x2.append(i)
        y2.append(PSO(w1=w1, a11=a11, a21=a21,w2=w1, a12=a11, a22=a21, dim=i,population_size=30, time_steps=1001, search_range=5.12, function=0))
    plt.plot(x, y, label = "Rastrigin function")
    plt.plot(x2, y2, label = "Sphere function")
    plt.xlabel('Number of dimensions')
    plt.ylabel('Best Swarm Performance')
    
    plt.title('Best Swarm Performance on different number of Search space dimensions')
    plt.legend()
    plt.show()

# comparing different swarms of PSO and their mixture
def plot_heterogeneous():
    x = []
    y = []
    x2 = []
    y2 = []
    x3 = []
    y3 = []
    # random set 1
    w1 = float(decimal.Decimal(random.randrange(0, 100))/100)
    a11 = float(decimal.Decimal(random.randrange(0, 250))/100)
    a21 = float(decimal.Decimal(random.randrange(0, 250))/100)
    # random set 2
    w2 = float(decimal.Decimal(random.randrange(0, 100))/100)
    a12 = float(decimal.Decimal(random.randrange(0, 250))/100)
    a22 = float(decimal.Decimal(random.randrange(0, 250))/100)

    for i in range(3,100):
        x.append(i)
        y.append(PSO(w1=0.75, a11=0.5, a21=0.5,w2=0.75, a12=0.5, a22=0.5, dim=i,population_size=30, time_steps=1001, search_range=5.12, function=1))
    for i in range(3,100):
        x2.append(i)
        y2.append(PSO(w1=0.85, a11=0.25, a21=0.25, w2=0.85, a12=0.25, a22=0.25, dim=i,population_size=30, time_steps=1001, search_range=5.12, function=1))
    for i in range(3,100):
        x3.append(i)
        y3.append(PSO(w1=0.75, a11=0.5, a21=0.5,w2=0.85, a12=0.25, a22=0.25, dim=i,population_size=30, time_steps=1001, search_range=5.12, function=1))
    # Compute the area using the composite trapezoidal rule.
    area1 = trapz(y, dx=1)
    area2 = trapz(y2, dx=1)
    area3 = trapz(y3, dx=1)
    print("Trapezoidal rule:\n")
    print("Swarm 1 area = " + str(area1))
    print("\nSwarm 2 area = " + str(area2))
    print("\nHeterogeneous area = " + str(area3))

    # Compute the area using the composite Simpson's rule.
    area1 = simpson(y, dx=1)
    area2 = simpson(y2, dx=1)
    area3 = simpson(y3, dx=1)
    print("\n\nSimpson's rule:\n")
    print("Swarm 1 area = " + str(area1))
    print("\nSwarm 2 area = " + str(area2))
    print("\nHeterogeneous area = " + str(area3))

    plt.ylabel('Best Swarm Performance')
    plt.xlabel('Number of Dimensions')
    label1 = "Swarm 1: α1=" + str(a11) + ", α2=" + str(a21) + " ω=" + str(w1)
    label2 = "Swarm 2: α1=" + str(a12) + ", α2=" + str(a22) + " ω=" + str(w2)
    plt.plot(x, y, label = label1)
    plt.plot(x2, y2, label = label2)
    plt.plot(x3, y3, label = "Heterogeneous - Swarm 1 & 2")
    plt.title('Best Swarm Performance on different number\n of Search space dimensions using different particle swarms')
    plt.legend()
    plt.show()

plot_heterogeneous()
# plot_extra_var()