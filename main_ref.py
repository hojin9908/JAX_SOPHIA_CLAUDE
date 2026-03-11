import numpy as np
import os, argparse
import matplotlib.pyplot as plt
import torch
import pickle
import time

from src_ref.gen_ptl import particle_generation
from src_ref.simulation_gpu2 import SPH_simulate_gpu
from visualize.animate import animate
from visualize.GUI import SPHVisualizer



def parsing():
    parser = argparse.ArgumentParser(description="Dam break SPH model run")

    # project property
    parser.add_argument("--tag", type=str, default="dam_break")
    parser.add_argument("--save_dir", type=str, default="./result")
    parser.add_argument("--gpu_num", type=int, help="gpu number to use", default=0)
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--den_cal_cuda", type=bool, help="Density calculation in cuda can take longer", default=False)

    # simulation setting
    parser.add_argument("--x", type=float, help="width of the dam", default=5)
    parser.add_argument("--y", type=float, help="height of the dam", default=3)
    parser.add_argument("--x0", type=float, help="width of initial particle distribution", default=2)
    parser.add_argument("--y0", type=float, help="height of initial particle distribution", default=1)
    parser.add_argument("--uniform", type=bool, help="Whether distribute initial particles uniformly", default=True)
    parser.add_argument("--Shepard", type=bool, help="Whether use Shepard filter for density calculation", default=True)
    parser.add_argument("--shepard_step", type=int, help="step interval for applying shepard filter", default=10)
    parser.add_argument("--search_all", type=bool, help="Search all partcles for weight calculation if True", default=True)
    parser.add_argument("--murnaghan", type=bool, help="Whether use Tait-Murnaghan pressure function", default=False)
    parser.add_argument("--history_save", type=bool, help="Whether save all history or not", default=False)
    
    # Physical coefficient
    parser.add_argument("--rho0", type=float, help="reference density", default=1000)
    parser.add_argument("--bnd_rho0", type=float, help="reference density for boundary", default=1000)
    parser.add_argument("--g", type=float, help="graviation acceleration", default=9.8)
    parser.add_argument("--c0", type=float, help="speed of sound", default=100)
    parser.add_argument("--gamma", type=float, help="stiffness parameter (1<=gamma<=7)", default=7)
    parser.add_argument("--K0", type=float, help="Bulk modulus", default=2.15e6)
    parser.add_argument("--p0", type=float, help="reference pressure", default=101325)

    # particle configuration hyperparameter
    parser.add_argument("--num_ptl", type=int, help="number of particle for simulation", default=1000)
    parser.add_argument("--bnd_layer", type=int, help="number of particle layer for dam boundary", default=3)
    parser.add_argument("--bnd_loss", type=float, help="hole width between boundary particles", default=1)
    parser.add_argument("--kappa", type=float, help="multiplication factor for support domain factor", default=3)

    # PDE solver hyperparameter
    parser.add_argument("--t", type=float, help="total time length of simulation", default=4)
    parser.add_argument("--dt", type=float, help="time step for simulation", default=1e-5)

    # Animation setting
    parser.add_argument("--interval", type=float, help="time interval between each image [ms]", default=10)
    parser.add_argument("--fps", type=int, help="frame per second for animation", default=20)
    parser.add_argument("--path", help="path for saving animation file", default="./result")
    parser.add_argument("--name", help="name of saving animation file", default="SPH_simulation")
    parser.add_argument("--ptl_s", type=float, help="size of particle in animation", default=0.1)
    parser.add_argument("--bnd_s", type=float, help="size of boundary particle in animation", default=0.1)
    parser.add_argument("--highlight_s", type=float, help="size of highlighted particle in animation", default=1)


    args = vars(parser.parse_args())

    return args


print("=============== Simulation start ===============")
if __name__ == "__main__":
    args = parsing()
    device = torch.device("cuda:{}".format(args['gpu_num']) if torch.cuda.is_available() else "cpu")

    a = time.time()
    init_model = particle_generation(ptl_num=args["num_ptl"],
                                     x=args["x"],
                                     y=args["y"],
                                     x0=args["x0"],
                                     y0=args["y0"],
                                     rho0=args["rho0"],
                                     bnd_layer=args["bnd_layer"],
                                     bnd_loss=args["bnd_loss"],
                                     uniform=args["uniform"],
                                     seed=args["seed"])
    bnd = init_model.dam2d_boundary()
    init = init_model.init_particle()

    SPH_model = SPH_simulate_gpu(boundary_ptls=bnd,
                             initial_states=init,
                             x=args["x"],
                             y=args["y"],
                             h=init_model.spacing,
                             kappa=args["kappa"],  
                             g=args["g"],
                             t=args["t"],
                             dt=args["dt"],
                             c0=args["c0"],
                             K0=args["K0"],
                             p0=args["p0"],
                             rho0=args["rho0"],
                             bnd_rho0=args["bnd_rho0"],
                             gamma=args["gamma"],
                             device=device,
                             density_cal_incuda=args["den_cal_cuda"],
                             shepard=args["Shepard"],
                             shepard_step=args["shepard_step"],
                             search_all=args["search_all"],
                             murnaghan=args["murnaghan"],
                             history_save=args["history_save"])
    SPH_model.simulation()
    b = time.time()
    print("Total Simulation ends in {}seconds".format(b-a))
    print("Total number of particle is {}".format(len(init)))
    animate_model = animate(data=SPH_model.trajectory,
                            bnd=bnd,
                            interval=args["interval"],
                            fps=args["fps"],
                            path=args["path"],
                            name=args["name"],
                            x=args["x"],
                            y=args["y"],
                            t=args["t"],
                            dt=args["dt"],
                            ptl_s=args["ptl_s"],
                            bnd_s=args["bnd_s"])
    animate_model.animation_create()

    # with open("./result/simulation_history.pkl", 'rb') as f:
    #     data = pickle.load(f)

    # GUI = SPHVisualizer(data=data,
    #                     x=args["x"],
    #                     y=args["y"],
    #                     ptl_size=args["ptl_s"],
    #                     highlight_size=args["highlight_s"])
    # GUI.mainloop()

                            
    