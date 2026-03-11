#%%
import numpy as np
import matplotlib.pyplot as plt


class particle_generation():
    def __init__(self, 
                 ptl_num : int = 1000, 
                 x : float = 5, 
                 y : float = 3, 
                 x0 : float = 2, 
                 y0 : float = 1, 
                 rho0 : float = 1000,
                 bnd_layer : int = 3,
                 bnd_loss : float = 1,
                 uniform : bool = True,
                 seed : int = 42):
        """
        creates 2d particle initial information
        ptl_num: number of particles for simulation
        x: width of dam
        y: height of dam
        x0: width of initial particle distribution
        y0: height of initial particle distribution
        rho0: reference density
        bnd_layer: number of particle layer for dam boundary
        bnd_loss: width between boundary particle
        uniform: distribute initial particle configuration uniformly(True) or randomly(False) 
        seed: random seed for initialization
        """
        self.ptl_num = ptl_num
        self.x = x
        self.y = y
        self.x0 = x0
        self.y0 = y0
        self.rho0 = rho0
        self.bnd_layer = bnd_layer
        self.bnd_loss = bnd_loss
        self.uniform = uniform
        self.seed = seed
        # calculate individual ptl's mass
        self.mass = self.rho0*self.x0*self.y0 / self.ptl_num


    def dam2d_boundary(self, plot = False):
        """
        creates 2d dam boundary particle informations
        result is 2d array. (# x 5)
        1st dim: ptl index
        2nd dim: x, y, vx, vy, m
        """
        self.spacing = np.sqrt(self.x0 * self.y0 / self.ptl_num)
        spacing = self.bnd_loss * self.spacing

        # left wall
        y1 = np.arange(0, self.y, spacing)
        x1 = []
        for i in range(self.bnd_layer):
            x1.append(np.zeros_like(y1) - (i+1) * spacing)

        xy_left = []
        for x_left in x1:
            xy_left.append(np.stack((x_left, y1), axis=1))

        left = xy_left[0]
        for i in range(1, len(xy_left)):
            left = np.concatenate((left, xy_left[i]), axis=0)

        # bottom wall
        x2 = np.arange(-self.bnd_layer*spacing, self.x + (self.bnd_layer+1)*spacing, spacing)
        y2 = []
        for i in range(self.bnd_layer):
            y2.append(np.zeros_like(x2) - (i+1) * spacing)

        xy_bottom = []
        for y_bottom in y2:
            xy_bottom.append(np.stack((x2, y_bottom), axis=1))

        bottom = xy_bottom[0]
        for i in range(1, len(xy_bottom)):
            bottom = np.concatenate((bottom, xy_bottom[i]), axis=0)

        # right wall
        x3 = []
        for i in range(self.bnd_layer):
            x3.append(np.zeros_like(y1) + self.x + (i+1) * spacing)

        xy_right = []
        for x_right in x3:
            xy_right.append(np.stack((x_right, y1), axis=1))

        right = xy_right[0]
        for i in range(1, len(xy_right)):
            right = np.concatenate((right, xy_right[i]), axis=0)

        # total
        boundary = np.concatenate((left, bottom, right), axis=0)
        boundary = np.concatenate((boundary, np.zeros_like(boundary)), axis=1)
        boundary = np.concatenate((boundary, np.zeros((len(boundary), 1))+self.mass), axis=1)
        print("total number of boundary particle: {:>6}".format(len(boundary)))

        if plot:
            scale = np.max([self.x, self.y])
            plt.xlim([-0.05 * scale, scale + 0.05 * scale])
            plt.ylim([-0.05 * scale, scale + 0.05 * scale])
            plt.scatter(boundary[:, 0], boundary[:, 1], s=0.1, c='black')
            plt.show()

        return boundary
    
    
    def init_particle(self, plot=False):
        """
        creates initial particles position and velocities
        result is 2d array. (# x 5)
        1st dim: ptl index
        2nd dim: x, y, vx, vy, m
        """
        if self.uniform:
            x_points = np.arange(0, self.x0, self.spacing)
            y_points = np.arange(0, self.y0, self.spacing)
            xx, yy = np.meshgrid(x_points, y_points)
            ptls = np.c_[xx.ravel(), yy.ravel()]
            init_ptls = np.concatenate((ptls, np.zeros_like(ptls)), axis = 1)
            self.ptl_num = len(init_ptls)
            self.mass = self.rho0*self.x0*self.y0 / self.ptl_num
            init_ptls = np.concatenate((init_ptls, np.zeros((len(init_ptls), 1))+self.mass), axis=1)

        else:
            np.random.seed(self.seed)
            xy_min = [0, 0]
            xy_max = [self.x0 , self.y0]
            ptls = np.random.uniform(low=xy_min, high=xy_max, size=(self.ptl_num,2))
            init_ptls = np.concatenate((ptls, np.zeros_like(ptls)), axis = 1)
            init_ptls = np.concatenate((init_ptls, np.zeros((len(init_ptls), 1))+self.mass), axis=1)

        print("total number of simulation particle: {:>2}".format(len(init_ptls)))
        
        if plot:
            boundary = self.dam2d_boundary()
            scale = np.max([self.x, self.y])
            plt.xlim([-0.05 * scale, scale + 0.05 * scale])
            plt.ylim([-0.05 * scale, scale + 0.05 * scale])
            plt.scatter(boundary[:, 0], boundary[:, 1], s=0.1, c='black')
            plt.scatter(init_ptls[:, 0], init_ptls[:, 1], s=0.2, c="red")
            plt.show()
        return init_ptls

# #%%
# model = particle_generation()
# model.init_particle(plot=True)
