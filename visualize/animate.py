#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
import os



class animate():
    def __init__(self,
                 data,
                 bnd,
                 interval = 10,
                 fps = 20,
                 path = "../result",
                 name = "SPH_simulation",
                 x = 5,
                 y = 3,
                 t = 1,
                 dt = 0.01,
                 ptl_s = 0.1,
                 bnd_s = 0.1):
        """
        Make gif animation of particle motion
        data: particles' motion data    [(time)x(#ptl)x(x,y,vx,vy,m)]
        bnd: boundary data      [(#ptl)x(x,y,vx,vy)]
        interval: time interval between each image [ms]
        fps: frame per second for animation
        path: path for saving file
        name: name of saving file
        x: width of dam
        y: height of dam
        ptl_s: size of particle
        bnd_s: size of boundary particle
        """
        self.data = data
        self.bnd = bnd
        self.interval = interval
        self.fps = int(1 / dt)
        self.path = path
        self.name = name
        self.x = x
        self.y = y
        self.t = t
        self.dt = dt
        self.ptl_s = ptl_s
        self.bnd_s = bnd_s
        self.interval = dt * 1000


    def create_frame(self, frame):
        plt.clf()
        scale = np.max([self.x, self.y])
        plt.xlim([-0.05 * scale, scale + 0.05 * scale])
        plt.ylim([-0.05 * scale, scale + 0.05 * scale])
        plt.scatter(self.data[frame,:,0], self.data[frame,:,1], s=self.ptl_s, c="blue")
        plt.scatter(self.bnd[:, 0], self.bnd[:, 1], s=self.bnd_s, c='black')
        plt.title("SPH simulation")


    def animation_create(self):
        fig = plt.figure()
        ani = animation.FuncAnimation(fig, self.create_frame, frames = range(0, len(self.data), int(1/(self.dt*100))), interval = self.interval)
        save_path = os.path.join(self.path, self.name+'.gif')
        ani.save(save_path, writer="imagemagick", fps=self.fps)


# #%%
# # debug
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../source')))
# from gen_ptl import particle_generation

# data = np.load("../result/simulation_result.npy")
# ptl_gen = particle_generation()
# bnd = ptl_gen.dam2d_boundary()
# animate_ptl = animate(data=data, bnd=bnd)
# animate_ptl.animation_create()