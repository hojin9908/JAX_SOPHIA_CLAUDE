#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import pickle

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../source')))
from gen_ptl import particle_generation
import kernel as kn


class SPH_simulate_gpu():
    def __init__(self,
                 boundary_ptls,
                 initial_states,
                 x=5,
                 y=3,
                 h=0.1,
                 kappa=3,
                 g=9.8, 
                 t=1,
                 dt=0.01,
                 c0=1500,
                 K0=2.15e6,
                 p0=101325,
                 rho0=1000,
                 bnd_rho0=1000,
                 gamma=7,
                 device='cpu',
                 density_cal_incuda=False,
                 shepard=False,
                 shepard_step=10,
                 search_all=True,
                 murnaghan=False,
                 history_save=False):
        """
        SPH simulate 2d dam break with initial particle state
        boundary_ptls: [(#bnd_ptl)x(x, y, vx, vy, m)]
        initial_states: [(#init_ptl)x(x, y, vx, vy, m)]
        x: width of dam
        y: height of dam
        h: support length
        kappa: kappa*h is support domain length
        g: gravity acceleration
        t: total time length of simulation
        dt: time step for simulation
        c0: speed of sound
        K0: Bulk modulus for Tait-Murnaghan pressure
        p0: reference density for Tait-Murnaghan pressure
        rho0: reference density
        bnd_rho0: reference density for boundary
        gamma: stiffness parameter (1<=gamma<=7)
        device: device for calculation "cpu" or "cuda"device
        density_cal_incuda=False: Density calculation in cuda device
        shepard: Execute Shepard filter for density calculation
        shepard_step: step interval for apply shepard filter
        search_all: search all particles' weight for True option.
        murnaghan: Use Tait_Murnaghan pressure function for pressure calculation
        history_save: Whether save all history or not
        """
        self.bnd = boundary_ptls
        self.ptl = initial_states
        self.x = x
        self.y = y
        self.h = h
        self.kappa = kappa
        self.g = g
        self.t = t
        self.dt = dt
        self.c0 = c0
        self.K0 = K0
        self.p0 = p0
        self.rho0 = rho0
        self.bnd_rho0 = bnd_rho0
        self.gamma = gamma
        self.device = device
        self.density_cal_incuda = density_cal_incuda
        self.shepard = shepard
        self.shepard_step = shepard_step
        self.search_all = search_all
        self.murnaghan = murnaghan
        self.history_save = history_save

        # initialize density
        self.ptl_rho = np.zeros((len(self.ptl), 1)) + self.rho0                     # [#ptl, 1]
        self.bnd_rho = np.zeros((len(self.bnd), 1)) + self.bnd_rho0                 # [#bnd, 1]
        # history of particel trajectory
        time_step = int(t/dt)
        self.trajectory = np.zeros((time_step, len(initial_states), 5))             # [time, #ptl, 5(x,y,vx,vy,m)]
        if self.history_save:
            self.ptl_rho_history = np.zeros((time_step, len(self.ptl), 1))          # [time, #ptl, 1]
            self.bnd_rho_history = np.zeros((time_step, len(self.bnd), 1))          # [time, #bnd, 1]
            self.ptl_pres_history = np.zeros((time_step, len(self.ptl), 1))         # [time, #ptl, 1]
            self.bnd_pres_history = np.zeros((time_step, len(self.bnd), 1))         # [time, #bnd, 1]
            self.ptl_pres_force_history = np.zeros((time_step, len(self.ptl), 2))   # [time, #ptl, 2]
            self.ptl_ptl_rho_weight_matrix_history = np.zeros((time_step, len(self.ptl), len(self.ptl)))        # [time, #ptl, #ptl]
            self.ptl_bnd_rho_weight_matrix_history = np.zeros((time_step, len(self.ptl), len(self.bnd)))        # [time, #ptl, #bnd]
            self.bnd_ptl_rho_weight_matrix_history = np.zeros((time_step, len(self.bnd), len(self.ptl)))        # [time, #bnd, #ptl]
            self.bnd_bnd_rho_weight_matrix_history = np.zeros((time_step, len(self.bnd), len(self.bnd)))        # [time, #bnd, #bnd]
            self.ptl_ptl_fpres_weight_matrix_history = np.zeros((time_step, len(self.ptl), len(self.ptl), 2))   # [time, #ptl, #ptl, 2]
            self.ptl_bnd_fpres_weight_matrix_history = np.zeros((time_step, len(self.ptl), len(self.bnd), 2))   # [time, #ptl, #bnd, 2]
        self.weight_init()
                                                                       


    def weights(self, subject, neighbor, function, dim=1):
        """
        return weight matrix between subject and neighbor with received function
        # input
        subject: subject ptls                               # [#ptl, 5(x,y,vx,vy,m)]
        neighbor: list of search range of neighbor ptls     # [#bnd, 5(x,y,vx,vy,m)]
        function: weight function
        dim: dimension of function output
        # output
        weights: matrix of weights                          # [#subject, #neighbor(, dim of output)]
        """
        # GPU parallelization
        if "cuda" in str(self.device):
            subject = torch.Tensor(subject).cuda()          # [#sub, 5]
            neighbor = torch.Tensor(neighbor).cuda()        # [#neig, 5]
            # Extract position information
            pos_sub = subject[:, :2].unsqueeze(1)           # [#sub, 1, 2]
            pos_neg = neighbor[:, :2].unsqueeze(0)          # [1, #neig, 2]
            # weights calculation
            weights = function(pos_sub, pos_neg, self.h)    # [#sub, #neig(, dim)]
            weights = weights.cpu()                         
        return weights
    

    def weight_init(self, function=kn.torch_Wendland2_2d_kernel):
        # Initialize weight
        # weight for ptl-ptl
        self.ptl_ptl_weight_matrix_rho = self.weights(subject=self.ptl,
                                                  neighbor=self.ptl,
                                                  function=function,
                                                  dim=1)                                                                          # [#ptl, #ptl]
        # weight for ptl-bnd
        self.ptl_bnd_weight_matrix_rho = self.weights(subject=self.ptl,
                                                  neighbor=self.bnd,
                                                  function=function,
                                                  dim=1)                                                                          # [#ptl, #bnd]
        # weight for bnd-ptl
        self.bnd_ptl_weight_matrix_rho = self.weights(subject=self.bnd,
                                                  neighbor=self.ptl,
                                                  function=function,
                                                  dim=1)                                                                          # [#bnd, #ptl]
        # weight for bnd-bnd
        self.bnd_bnd_weight_matrix_rho = self.weights(subject=self.bnd,
                                                  neighbor=self.bnd,
                                                  function=function,
                                                  dim=1)             


    def shepard_filter(self):
        """
        Calculate Shepard filter
        filter_i = sum[(m/rho)Wij]
        """
        m_ptl = torch.Tensor(self.ptl[:, -1])               # [#ptl]
        m_bnd = torch.Tensor(self.bnd[:, -1])               # [#bnd]
        rho_ptl = torch.Tensor(self.ptl_rho)                # [#ptl, 1]
        rho_bnd = torch.Tensor(self.bnd_rho)                # [#bnd, 1]
        self.weight_init()
        # Calculate filter for ptl
        ptl_ptl_filter = self.ptl_ptl_weight_matrix_rho * m_ptl.expand_as(self.ptl_ptl_weight_matrix_rho) / rho_ptl.t().expand_as(self.ptl_ptl_weight_matrix_rho)   # [#ptl, #ptl]
        ptl_bnd_filter = self.ptl_bnd_weight_matrix_rho * m_bnd.expand_as(self.ptl_bnd_weight_matrix_rho) / rho_bnd.t().expand_as(self.ptl_bnd_weight_matrix_rho)   # [#ptl, #bnd]
        self.ptl_filter = ptl_ptl_filter.sum(dim=1) + ptl_bnd_filter.sum(dim=1)  # [#ptl]   
        # Calculater filter for bnd
        bnd_ptl_filter = self.bnd_ptl_weight_matrix_rho * m_ptl.expand_as(self.bnd_ptl_weight_matrix_rho) / rho_ptl.t().expand_as(self.bnd_ptl_weight_matrix_rho) 
        bnd_bnd_filter = self.bnd_bnd_weight_matrix_rho * m_bnd.expand_as(self.bnd_bnd_weight_matrix_rho) / rho_bnd.t().expand_as(self.bnd_bnd_weight_matrix_rho)
        self.bnd_filter = bnd_ptl_filter.sum(dim=1) + bnd_bnd_filter.sum(dim=1)  # [#bnd]


    def weights_matrix_rho(self, function=kn.torch_Wendland2_2d_kernel, dim=1):
        """
        Calculate Wij for density calculation
        self.ptl_filter                         # [#ptl]
        self.bnd_filter                         # [#bnd]
        """
        # weight for ptl-ptl
        self.ptl_ptl_weight_matrix_rho = self.weights(subject=self.ptl,
                                                  neighbor=self.ptl,
                                                  function=function,
                                                  dim=dim)                                                                          # [#ptl, #ptl]
        self.ptl_ptl_weight_matrix_rho = self.ptl_ptl_weight_matrix_rho / self.ptl_filter.unsqueeze(1).expand_as(self.ptl_ptl_weight_matrix_rho) # [#ptl, #ptl]
        # weight for ptl-bnd
        self.ptl_bnd_weight_matrix_rho = self.weights(subject=self.ptl,
                                                  neighbor=self.bnd,
                                                  function=function,
                                                  dim=dim)                                                                          # [#ptl, #bnd]
        self.ptl_bnd_weight_matrix_rho = self.ptl_bnd_weight_matrix_rho / self.ptl_filter.unsqueeze(1).expand_as(self.ptl_bnd_weight_matrix_rho) # [#ptl, #bnd]
        # weight for bnd-ptl
        self.bnd_ptl_weight_matrix_rho = self.weights(subject=self.bnd,
                                                  neighbor=self.ptl,
                                                  function=function,
                                                  dim=dim)                                                                          # [#bnd, #ptl]
        self.bnd_ptl_weight_matrix_rho = self.bnd_ptl_weight_matrix_rho / self.bnd_filter.unsqueeze(1).expand_as(self.bnd_ptl_weight_matrix_rho) # [#bnd, #ptl]
        # weight for bnd-bnd
        self.bnd_bnd_weight_matrix_rho = self.weights(subject=self.bnd,
                                                  neighbor=self.bnd,
                                                  function=function,
                                                  dim=dim)                                                                          # [#bnd, #bnd]
        self.bnd_bnd_weight_matrix_rho = self.bnd_bnd_weight_matrix_rho / self.bnd_filter.unsqueeze(1).expand_as(self.bnd_bnd_weight_matrix_rho) # [#bnd, #bnd]
        return
    

    def weights_matrix_fpres(self, function=kn.torch_grad2d_Wendland2_kernel, dim=1):
        """
        Calculate grad_Wij for pressure force calculation
        self.ptl_filter                         # [#ptl]
        self.bnd_filter                         # [#bnd]
        """
        # weight for ptl-ptl
        self.ptl_ptl_weight_matrix_fpres = self.weights(subject=self.ptl,
                                                  neighbor=self.ptl,
                                                  function=function,
                                                  dim=dim)                                                                                                            # [#ptl, #ptl, 2]
        self.ptl_ptl_weight_matrix_fpres = self.ptl_ptl_weight_matrix_fpres / self.ptl_filter.unsqueeze(1).unsqueeze(2).expand_as(self.ptl_ptl_weight_matrix_fpres)   # [#ptl, #ptl, 2]
        # weight for ptl-bnd
        self.ptl_bnd_weight_matrix_fpres = self.weights(subject=self.ptl,
                                                  neighbor=self.bnd,
                                                  function=function,
                                                  dim=dim)                                                                                                            # [#ptl, #bnd, 2]
        self.ptl_bnd_weight_matrix_fpres = self.ptl_bnd_weight_matrix_fpres / self.ptl_filter.unsqueeze(1).unsqueeze(2).expand_as(self.ptl_bnd_weight_matrix_fpres)   # [#ptl, #bnd, 2]
        # weight for bnd-ptl
        self.bnd_ptl_weight_matrix_fpres = self.weights(subject=self.bnd,
                                                  neighbor=self.ptl,
                                                  function=function,
                                                  dim=dim)                                                                                                            # [#bnd, #ptl, 2]
        self.bnd_ptl_weight_matrix_fpres = self.bnd_ptl_weight_matrix_fpres / self.bnd_filter.unsqueeze(1).unsqueeze(2).expand_as(self.bnd_ptl_weight_matrix_fpres)   # [#bnd, #ptl, 2]
        # weight for bnd-bnd
        self.bnd_bnd_weight_matrix_fpres = self.weights(subject=self.bnd,
                                                  neighbor=self.bnd,
                                                  function=function,
                                                  dim=dim)                                                                                                            # [#bnd, #bnd, 2]
        self.bnd_bnd_weight_matrix_fpres = self.bnd_bnd_weight_matrix_fpres / self.bnd_filter.unsqueeze(1).unsqueeze(2).expand_as(self.bnd_bnd_weight_matrix_fpres)   # [#bnd, #ptl, 2]
        return


    def density_cal_search_all(self):
        """
        Calculate density
        """
        # GPU parallelization
        if ("cuda" in str(self.device)) and (self.device):
            m_ptl = torch.Tensor(self.ptl[:, 4])                                                            # [#ptl]
            m_bnd = torch.Tensor(self.bnd[:, 4])                                                            # [#bnd]
            # Calculate ptl density
            ptl_ptl_rho = self.ptl_ptl_weight_matrix_rho * m_ptl.expand_as(self.ptl_ptl_weight_matrix_rho)  # [#ptl, #ptl]
            ptl_bnd_rho = self.ptl_bnd_weight_matrix_rho * m_bnd.expand_as(self.ptl_bnd_weight_matrix_rho)  # [#ptl, #bnd]
            ptl_rho = ptl_ptl_rho.sum(dim=1) + ptl_bnd_rho.sum(dim=1)                                       # [#ptl]
            ptl_rho = ptl_rho.unsqueeze(-1).cpu().numpy()                                                   # [#ptl, 1]
            # Calculate bnd density
            bnd_ptl_rho = self.bnd_ptl_weight_matrix_rho * m_ptl.expand_as(self.bnd_ptl_weight_matrix_rho)  # [#bnd, #ptl]
            bnd_bnd_rho = self.bnd_bnd_weight_matrix_rho * m_bnd.expand_as(self.bnd_bnd_weight_matrix_rho)  # [#bnd, #bnd]
            bnd_rho = bnd_ptl_rho.sum(dim=1) + bnd_bnd_rho.sum(dim=1)                                       # [#bnd]
            bnd_rho = bnd_rho.unsqueeze(-1).cpu().numpy()                                                   # [#bnd, 1]
            # Update
            self.ptl_rho = ptl_rho
            self.bnd_rho = bnd_rho
        else:
            raise NotImplementedError("Do not try this without GPU")
        return
    

    def pres_cal(self):
        """
        Calculate pressure of each particle with Equation of State(EOS)
        # input
        self.ptl_rho                                    # [#ptl, 1]
        self.bnd_rho                                    # [#bnd, 1]
        # output
        self.ptl_pres                                   # [#ptl, 1]
        self.bnd_pres                                   # [#bnd, 1]
        """
        self.ptl_pres = np.zeros((len(self.ptl), 1))    # [#ptl, 1]
        self.bnd_pres = np.zeros((len(self.bnd), 1))    # [#bnd, 1]
        # Calculate pressure with EOS
        for i in range(len(self.ptl)):
            self.ptl_pres[i] = self.pres_func(rho=self.ptl_rho[i])
        for i in range(len(self.bnd)):
            self.bnd_pres[i] = self.pres_func(rho=self.bnd_rho[i])
        return
            

    def pres_force_cal(self):
        """
        Calculate acceleration by pressure force
        note: r_ij = xi -xj is used
        """
        # GPU parallelization
        if ("cuda" in str(self.device)) and self.search_all:
            m_ptl = torch.Tensor(self.ptl[:, 4]).unsqueeze(0).unsqueeze(2)  # [1, #ptl, 1]
            m_bnd = torch.Tensor(self.bnd[:, 4]).unsqueeze(0).unsqueeze(2)  # [1, #bnd, 1]
            rho_ptl = torch.Tensor(self.ptl_rho).unsqueeze(0)               # [1, #ptl, 1]
            rho_bnd = torch.Tensor(self.bnd_rho).unsqueeze(0)               # [1, #bnd, 1]
            pres_ptl = torch.Tensor(self.ptl_pres).unsqueeze(0)             # [1, #ptl, 1]
            pres_bnd = torch.Tensor(self.bnd_pres).unsqueeze(0)             # [1, #bnd, 1]
            pres_i = torch.Tensor(self.ptl_pres).unsqueeze(1)               # [#ptl, 1, 1]
            # Calculate pressure force (only for ptl)
            ptl_ptl_pres_force = self.ptl_ptl_weight_matrix_fpres \
                * m_ptl.expand_as(self.ptl_ptl_weight_matrix_fpres) \
                * (pres_ptl.expand_as(self.ptl_ptl_weight_matrix_fpres) + pres_i.expand_as(self.ptl_ptl_weight_matrix_fpres)) \
                / rho_ptl.expand_as(self.ptl_ptl_weight_matrix_fpres)
            ptl_bnd_pres_force = self.ptl_bnd_weight_matrix_fpres \
                * m_bnd.expand_as(self.ptl_bnd_weight_matrix_fpres) \
                * (pres_bnd.expand_as(self.ptl_bnd_weight_matrix_fpres) + pres_i.expand_as(self.ptl_bnd_weight_matrix_fpres))\
                / rho_bnd.expand_as(self.ptl_bnd_weight_matrix_fpres)
            ptl_pres_force = ptl_ptl_pres_force.sum(dim=1) + ptl_bnd_pres_force.sum(dim=1)  # [#ptl, 2]
            ptl_pres_force = -ptl_pres_force / rho_ptl.squeeze().unsqueeze(1).expand_as(ptl_pres_force)  # [#ptl, 2]
            self.ptl_pres_force = ptl_pres_force.cpu().numpy()
        return 
    

    def gravity_force_cal(self):
        """
        Calculate gravity acceleration
        """
        self.ptl_grav_force = np.zeros((len(self.ptl), 2))                  # [#ptl, 2]
        self.ptl_grav_force[:, 1] = self.ptl_grav_force[:, 1] - self.g      # [#ptl, 2]
        return


    def vel_step(self):
        """
        From Navier-Stokes equation, update velocity
        self.ptl        [#ptl, 5(x,y,vx,vy,m)]
        """
        self.ptl[:, 2:4] = self.ptl[:, 2:4] + (self.ptl_grav_force + self.ptl_pres_force)*self.dt
        return
    

    def pos_step(self):
        """
        Update position of particles
        self.ptl        [#ptl, 5(x,y,vx,vy,m)]
        """
        self.ptl[:, :2] = self.ptl[:, :2] + self.ptl[:, 2:4]*self.dt
        return
    

    def step(self, step):
        """
        Run a single step of SPH simulation
        """
        print("----------------{}th step calculation----------------".format(step))
        self.trajectory[step] = self.ptl                                # [#ptl, 5(x,y,vx,vy,m)]
        # density calculation
        a = time.time()
        if (step) % self.shepard_step == 0:
            self.shepard_filter()
        self.weights_matrix_rho()
        self.density_cal_search_all()
        if self.history_save:
            self.ptl_rho_history[step] = self.ptl_rho                                       # [#ptl, 1]
            self.bnd_rho_history[step] = self.bnd_rho                                       # [#bnd, 1]
            self.ptl_ptl_rho_weight_matrix_history[step] = self.ptl_ptl_weight_matrix_rho   # [#ptl, #ptl]
            self.ptl_bnd_rho_weight_matrix_history[step] = self.ptl_bnd_weight_matrix_rho   # [#ptl, #bnd]
            self.bnd_ptl_rho_weight_matrix_history[step] = self.bnd_ptl_weight_matrix_rho   # [#bnd, #ptl]
            self.bnd_bnd_rho_weight_matrix_history[step] = self.bnd_bnd_weight_matrix_rho   # [#bnd, #bnd]
        b = time.time()
        print("density_cal one step end in {:.3f}seconds".format(b-a))
        # pressure calculation
        a = time.time()
        self.pres_cal()
        if self.history_save:
            self.ptl_pres_history[step] = self.ptl_pres                # [#ptl, 1]
            self.bnd_pres_history[step] = self.bnd_pres                # [#bnd, 1]
        b = time.time()
        print("pres_cal one step end in {:.3f}seconds".format(b-a))
        # pressure force calculation
        a = time.time()
        self.weights_matrix_fpres(function=kn.torch_grad2d_Wendland2_kernel, dim=2)
        self.pres_force_cal()
        if self.history_save:
            self.ptl_pres_force_history[step] = self.ptl_pres_force                             # [#ptl, 2]
            self.ptl_ptl_fpres_weight_matrix_history[step] = self.ptl_ptl_weight_matrix_fpres   # [#ptl, #ptl, 2]
            self.ptl_bnd_fpres_weight_matrix_history[step] = self.ptl_bnd_weight_matrix_fpres   # [#ptl, #bnd, 2]
        b = time.time()
        print("pres__force_cal one step end in {:.3f}seconds".format(b-a))
        # gravity force calculation
        a = time.time()
        self.gravity_force_cal()
        b = time.time()
        print("pres_gravity_force_cal one step end in {:.3f}seconds".format(b-a))
        # velocity update
        a = time.time()
        self.vel_step()
        b = time.time()
        print("vel_step one step end in {:.3f}seconds".format(b-a))
        # position update
        a = time.time()
        self.pos_step()
        b = time.time()
        print("pos_step one step end in {:.3f}seconds".format(b-a))
        print("-----------------------------------------------------")
        return
    

    def simulation(self, each_step_plot=False):
        """
        Run a total time simulation
        """
        # Choose Equation of State(EOS) function
        if self.murnaghan:
            self.pres_func = self.tait_murnaghan_p
        else:
            self.pres_func = self.tait_p

        # Simulation start
        total_step = len(self.trajectory)
        for t in range(total_step):
            if each_step_plot:
                self.plot_ptl()
            self.step(step=t)

        # Save the result
        np.save("./result/simulation_trajectory.npy", self.trajectory)
        if self.history_save:
            simulation_history = {
                "trajectory": self.trajectory,                          # [time, #ptl, 5(x,y,vx,vy,m)]
                "boundary": self.bnd,                                   # [#bnd, 5(x,y,vx,vy,m)]
                "ptl_rho_history": self.ptl_rho_history,                # [time, #ptl, 1]
                "bnd_rho_history": self.bnd_rho_history,                # [time, #bnd, 1]
                "ptl_pres_history": self.ptl_pres_history,              # [time, #ptl, 1]
                "bnd_pres_history": self.bnd_pres_history,              # [time, #bnd, 1]
                "ptl_pres_force_history": self.ptl_pres_force_history,  # [time, #ptl, 2]
                "ptl_ptl_rho_weight_matrix_history": self.ptl_ptl_rho_weight_matrix_history,        # [time, #ptl, #ptl]
                "ptl_bnd_rho_weight_matrix_history": self.ptl_bnd_rho_weight_matrix_history,        # [time, #ptl, #bnd]
                "bnd_ptl_rho_weight_matrix_history": self.bnd_ptl_rho_weight_matrix_history,        # [time, #bnd, #ptl]
                "bnd_bnd_rho_weight_matrix_history": self.bnd_bnd_rho_weight_matrix_history,        # [time, #bnd, #bnd]
                "ptl_ptl_fpres_weight_matrix_history": self.ptl_ptl_fpres_weight_matrix_history,    # [time, #ptl, #ptl, 2]
                "ptl_bnd_fpres_weight_matrix_history": self.ptl_bnd_fpres_weight_matrix_history,    # [time, #ptl, #bnd, 2]
                "t": self.t,                                            # [1]
                "dt": self.dt,                                          # [1]
                "x": self.x,                                            # [1]
                "y": self.y                                             # [1]
            }
            with open("./result/simulation_history.pkl", 'wb') as f:
                pickle.dump(simulation_history, f)
        return


    def plot_ptl(self):
        scale = np.max([self.x, self.y])
        plt.xlim([-0.05 * scale, scale + 0.05 * scale])
        plt.ylim([-0.05 * scale, scale + 0.05 * scale])
        plt.scatter(self.bnd[:, 0], self.bnd[:, 1], s=0.05, c='black')
        plt.scatter(self.ptl[:, 0], self.ptl[:, 1], s=0.05, c="red")
        plt.show()
        return


    def tait_p(self,rho):
        """
        calculate pressure from the density with Tait equation
        rho: density
        """
        const = self.c0**2 * self.rho0 / self.gamma
        main = (rho/self.rho0)**self.gamma - 1
        return const*main
    
    
    def tait_murnaghan_p(self, rho):
        """
        Calculate pressure from the density with Tait-Murnaghan equation
        rho: density
        """
        const = self.K0/self.gamma
        main = (rho/self.rho0)**self.gamma - 1
        return const*main + self.p0



# #######################################################################################
# #%%
# #debug
# ptl_gen = particle_generation()
# bnd = ptl_gen.dam2d_boundary()
# init = ptl_gen.init_particle(plot=True)
# model = SPH_simulate_gpu(boundary_ptls=bnd, initial_states=init, h=ptl_gen.spacing)
# xi = torch.Tensor(init[:, :2]).unsqueeze(1)
# xj = torch.Tensor(init[:, :2]).unsqueeze(0)
# weight = kn.torch_grad2d_Wendland2_kernel(xi=xi, xj=xj, h=model.h)
# h = model.h
# r_vec = xi - xj                             # [#sub, #neig, 2]
# r = torch.linalg.norm(xi-xj, dim=-1)        # [#sub, #neig]
# r = torch.stack((r,r), dim=-1)              # [#sub, #neig, 2]
