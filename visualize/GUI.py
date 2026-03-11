#%%
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import pickle
import numpy as np
import os

class SPHVisualizer(tk.Tk):
    def __init__(self, data, ptl_size=0.3, highlight_size=2):
        super().__init__()
        """
        x: width of dam
        y: height of dam
        ptl_size: size of particle in simulation
        highlight_size: size of particle in concern in simulation
        """
        self.x = data["x"]
        self.y = data["y"]
        self.t = data["t"]
        self.dt = data["dt"]
        self.ptl_size = ptl_size
        self.highlight_size = highlight_size
        self.title("SPH Simulation Visualizer")
        self.geometry("1400x1000")
        self.one_frame = int(self.t/(self.dt*100))

        # Simulation data
        self.trajectory = data['trajectory']                            # [(time)x(#ptl)x(x, y, vx, vy, m)]
        self.boundary = data['boundary']                                # [(#bnd)x(x, y, vx, vy, m)]
        self.ptl_rho_history = data['ptl_rho_history']                  # [(time)x(#ptl)x1]
        self.bnd_rho_history = data['bnd_rho_history']                  # [(time)x(#bnd)x1]
        self.ptl_pres_history = data['ptl_pres_history']                # [(time)x(#ptl)x1]
        self.bnd_pres_history = data['bnd_pres_history']                # [(time)x(#bnd)x1]
        self.ptl_pres_force_history = data['ptl_pres_force_history']    # [(time)x(#ptl)x2]
        self.ptl_ptl_rho_weight_matrix_history = data["ptl_ptl_rho_weight_matrix_history"]
        self.ptl_bnd_rho_weight_matrix_history = data["ptl_bnd_rho_weight_matrix_history"]
        self.bnd_ptl_rho_weight_matrix_history = data["bnd_ptl_rho_weight_matrix_history"]
        self.bnd_bnd_rho_weight_matrix_history = data["bnd_bnd_rho_weight_matrix_history"]
        # rho weight
        shape = self.ptl_ptl_rho_weight_matrix_history.shape
        bnd_shape = self.bnd_bnd_rho_weight_matrix_history.shape
        rho_weight_matrix_history = np.zeros((shape[0], shape[1]+bnd_shape[1], shape[1]+bnd_shape[1]))
        rho_weight_matrix_history[:, :shape[1], :shape[1]] = self.ptl_ptl_rho_weight_matrix_history
        rho_weight_matrix_history[:, :shape[1], shape[1]:] = self.ptl_bnd_rho_weight_matrix_history
        rho_weight_matrix_history[:, shape[1]:, :shape[1]] = self.bnd_ptl_rho_weight_matrix_history
        rho_weight_matrix_history[:, shape[1]:, shape[1]:] = self.bnd_bnd_rho_weight_matrix_history
        self.rho_weight_matrix_history = rho_weight_matrix_history
        # fpres weight
        self.ptl_ptl_fpres_weight_matrix_history = data["ptl_ptl_fpres_weight_matrix_history"]
        self.ptl_bnd_fpres_weight_matrix_history = data["ptl_bnd_fpres_weight_matrix_history"]
        fpres_weight_matrix_history = np.zeros((shape[0], shape[1], shape[1]+bnd_shape[1], 2))
        fpres_weight_matrix_history[:, :, :shape[1], :] = self.ptl_ptl_fpres_weight_matrix_history
        fpres_weight_matrix_history[:, :, shape[1]:, :] = self.ptl_bnd_fpres_weight_matrix_history
        self.fpres_weight_matrix_history = fpres_weight_matrix_history
        
        # Left upper side for animated trajectory plot
        self.trajectory_frame = ttk.Frame(self)
        self.trajectory_frame.grid(row=0, column=0, sticky="nsew")
        self.scatter_particles = None  # Scatter plot for particles
        self.scatter_boundary = None   # Scatter plot for boundaries
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.frame = 0
        self.highlighted_particle_index = None
        self.highlighted_boundary_index = None

        # Right upper side for information graph
        self.graph_frame = ttk.Frame(self)
        self.graph_frame.grid(row=0, column=1, sticky="nsew")
            # Create a frame for the plot area within graph_frame
        self.plot_container = ttk.Frame(self.graph_frame)
        self.plot_container.pack(expand=True, fill=tk.BOTH)

        # Right right upper side for user input
        self.input_frame = ttk.Frame(self)
        self.input_frame.grid(row=0, column=2, sticky="nsew")
            # ptl or bnd choose
        self.choice_label = ttk.Label(self.input_frame, text="Choose Data Type:")
        self.choice_label.pack()
        self.data_type = tk.StringVar(value="Particle")
        self.data_type_dropdown = ttk.Combobox(self.input_frame, textvariable=self.data_type, values=["Particle", "Boundary"], state="readonly")
        self.data_type_dropdown.pack()
            # index
        self.index_label = ttk.Label(self.input_frame, text="Enter Index:")
        self.index_label.pack()
            # input window
        self.index_entry = ttk.Entry(self.input_frame)
        self.index_entry.pack()
            # enter data button
        self.show_button = ttk.Button(self.input_frame, text="Show Data", command=self.plot_data)
        self.show_button.pack()

        # Left lower side for animated weight plot
        self.weight_frame = ttk.Frame(self)
        self.weight_frame.grid(row=1, column=0, sticky="nsew")
        self.fig_weight, self.ax_weight = plt.subplots(figsize=(5, 5))

        # Right lower side for animated weight plot
        self.xpres_weight_frame = ttk.Frame(self)
        self.xpres_weight_frame.grid(row=1, column=1, sticky="nsew")
        self.fig_xpres_weight, self.ax_xpres_weight = plt.subplots(figsize=(5, 5))

        self.init_plot()
        self.init_weight_plot()
        self.animate_trajectory()
    


    def init_weight_plot(self):
        self.weight = self.ax_weight.matshow(self.rho_weight_matrix_history[0])
        self.weight_xpres = self.ax_xpres_weight.matshow(self.fpres_weight_matrix_history[0,:,:,0])

        self.ax_weight.set_title("weights")
        self.ax_xpres_weight.set_title("Pressure x acceleration weights")

        self.fig_weight.colorbar(self.weight)
        self.fig_xpres_weight.colorbar(self.weight_xpres)

        self.canvas_weight = FigureCanvasTkAgg(self.fig_weight, master=self.weight_frame)
        self.canvas_weight_xpres = FigureCanvasTkAgg(self.fig_xpres_weight, master=self.xpres_weight_frame)

        self.canvas_weight.draw()
        self.canvas_weight_xpres.draw()

        self.canvas_weight.get_tk_widget().pack()
        self.canvas_weight_xpres.get_tk_widget().pack()

    def animate_weight_plot(self):
        self.weight.set_data(self.rho_weight_matrix_history[self.frame])
        self.weight_xpres.set_data(self.fpres_weight_matrix_history[self.frame,:,:,0])

        self.canvas_weight.draw()
        self.canvas_weight_xpres.draw()
        


    def init_plot(self):
        """Initialize the scatter plots and axis limits."""
        scale = max(self.x, self.y)
        self.ax.set_xlim([-0.05 * scale, scale + 0.05 * scale])
        self.ax.set_ylim([-0.05 * scale, scale + 0.05 * scale])
        # Initialize scatter plot with the first frame's data
        self.scatter_particles = self.ax.scatter(self.trajectory[0, :, 0], self.trajectory[0, :, 1], s=self.ptl_size, c="blue", label="Particles")
        self.scatter_boundary = self.ax.scatter(self.boundary[:, 0], self.boundary[:, 1], s=self.ptl_size, c="black", label="Boundary")
        self.ax.set_title("SPH Simulation - Particle Trajectory")
        self.ax.legend()
        # Embed the plot into tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.trajectory_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

    def animate_trajectory(self):
        """Update the scatter plot with the next frame's data."""
        self.frame += self.one_frame # Move to the next frame
        if self.frame >= len(self.trajectory):
            self.frame = 0  # Loop back to the first frame
        # Update particle positions
        self.scatter_particles.set_offsets(self.trajectory[self.frame, :, :2])
        # Update boundary positions
        self.scatter_boundary.set_offsets(self.boundary[:, :2])
        # Update particle colors
        self.update_particle_colors()
        # Redraw the canvas
        self.canvas.draw()
        # Update the vertical line in the right-side plot
        self.update_graph_line()
        self.animate_weight_plot()
        # Schedule the next update
        self.after(10, self.animate_trajectory)  # Update every 10 ms (10 FPS)

    def highlight(self, index, data_type):
        """Highlight a specific particle or boundary in red."""
        if data_type == "Particle":
            self.highlighted_particle_index = index
            self.highlighted_boundary_index = None
        elif data_type == "Boundary":
            self.highlighted_particle_index = None
            self.highlighted_boundary_index = index
        self.update_particle_colors()

    def update_particle_colors(self):
        """Update the color of particles and boundaries based on the highlighted index."""
        if self.highlighted_particle_index is not None:
            # Update the color for the highlighted particle
            positions = self.trajectory[self.frame, :, :2]
            colors = np.array(['red' if i == self.highlighted_particle_index else 'blue'
                            for i in range(len(positions))])
            sizes = np.array([self.highlight_size if i == self.highlighted_particle_index else self.ptl_size
                          for i in range(len(positions))])
            self.scatter_particles.set_edgecolor(colors)
            self.scatter_particles.set_sizes(sizes)
        else:
            self.scatter_particles.set_edgecolor('blue')
        
        if self.highlighted_boundary_index is not None:
            # Update the color for the highlighted boundary
            boundary_positions = self.boundary[:, :2]
            boundary_colors = np.array(['red' if i == self.highlighted_boundary_index else 'black'
                                        for i in range(len(boundary_positions))])
            boundary_sizes = np.array([self.highlight_size if i == self.highlighted_boundary_index else self.ptl_size
                          for i in range(len(boundary_positions))])
            self.scatter_boundary.set_edgecolor(boundary_colors)
            self.scatter_boundary.set_sizes(boundary_sizes)
        else:
            self.scatter_boundary.set_edgecolor('black')



    def plot_data(self):
        """Plot data for either particle or boundary based on user selection."""
        data_type = self.data_type.get()  # Either "Particle" or "Boundary"
        try:
            index = int(self.index_entry.get())
            time = self.dt*np.arange(self.ptl_rho_history.shape[0])
            self.highlight(index, data_type)

            for widget in self.plot_container.winfo_children():
                widget.destroy() 

            fig, axs = plt.subplots(4, figsize=(7, 5))
            if data_type == "Particle":
                # Plot rho history for particles
                axs[0].plot(time, self.ptl_rho_history[:, index, 0])
                axs[0].set_title(f'Particle {index} - Density (Rho)')
                axs[0].set_xlabel('Time')
                axs[0].set_ylabel('Density')
                # Plot pressure history for particles
                axs[1].plot(time, self.ptl_pres_history[:, index, 0])
                axs[1].set_title(f'Particle {index} - Pressure')
                axs[1].set_xlabel('Time')
                axs[1].set_ylabel('Pressure')
                # Plot pressure force components for particles
                axs[2].plot(time, self.ptl_pres_force_history[:, index, 0], label="Pressure a_x")
                axs[2].plot(time, self.ptl_pres_force_history[:, index, 1], label="Pressure a_y")
                axs[2].set_title(f'Particle {index} - Pressure Force Acceleration Components')
                axs[2].set_xlabel('Time')
                axs[2].set_ylabel('Acceleration')
                axs[2].legend()
                # Plot velocity components for particles
                axs[3].plot(time, self.trajectory[:, index, 2], label="v_x")
                axs[3].plot(time, self.trajectory[:, index, 3], label="v_y")
                axs[3].set_title(f'Particle {index} - Velocity Components')
                axs[3].set_xlabel('Time')
                axs[3].set_ylabel('Velocity')
                axs[3].legend()
            elif data_type == "Boundary":
                # Plot rho history for boundary particles
                axs[0].plot(time, self.bnd_rho_history[:, index, 0])
                axs[0].set_title(f'Boundary {index} - Density (Rho)')
                axs[0].set_xlabel('Time')
                axs[0].set_ylabel('Density')
                # Plot pressure history for boundary particles
                axs[1].plot(time, self.bnd_pres_history[:, index, 0])
                axs[1].set_title(f'Boundary {index} - Pressure')
                axs[1].set_xlabel('Time')
                axs[1].set_ylabel('Pressure')
            plt.subplots_adjust(hspace=1)
            canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
            canvas.draw()
            canvas.get_tk_widget().pack()
            self.right_fig = fig
            self.right_axs = axs
        except ValueError:
            print("Invalid input. Please enter a valid index.")
        except IndexError:
            print("Index out of range. Please try another index.")
            
    def update_graph_line(self):
        """Update the vertical line in the right-side plot to match the left-side animation time."""
        if hasattr(self, 'right_axs'):
            # Remove old vertical lines (if any)
            for ax in self.right_axs:
                for line in ax.lines:
                    if line.get_linestyle() == '--':  # Assuming the line to remove is dashed
                        line.remove()
            # Add new vertical lines at current frame time
            for ax in self.right_axs:
                ax.axvline(x=self.frame * self.dt, color='grey', linestyle='--')  # Scale frame time by time-step (0.01s)
            # Redraw the figure
            self.right_fig.canvas.draw()



########################################################################################################################################################
########################################################################################################################################################



class SPHVisualizer_no_weight(tk.Tk):
    def __init__(self, data, ptl_size=0.3, highlight_size=2):
        super().__init__()
        """
        x: width of dam
        y: height of dam
        ptl_size: size of particle in simulation
        highlight_size: size of particle in concern in simulation
        """
        self.x = data["x"]
        self.y = data["y"]
        self.t = data["t"]
        self.dt = data["dt"]
        self.ptl_size = ptl_size
        self.highlight_size = highlight_size
        self.title("SPH Simulation Visualizer")
        self.geometry("1400x900")

        # Simulation data
        self.trajectory = data['trajectory']                            # [(time)x(#ptl)x(x, y, vx, vy, m)]
        self.boundary = data['boundary']                                # [(#bnd)x(x, y, vx, vy, m)]
        self.ptl_rho_history = data['ptl_rho_history']                  # [(time)x(#ptl)x1]
        self.bnd_rho_history = data['bnd_rho_history']                  # [(time)x(#bnd)x1]
        self.ptl_pres_history = data['ptl_pres_history']                # [(time)x(#ptl)x1]
        self.bnd_pres_history = data['bnd_pres_history']                # [(time)x(#bnd)x1]
        self.ptl_pres_force_history = data['ptl_pres_force_history']    # [(time)x(#ptl)x2]
        
        # Left side for animated trajectory plot
        self.trajectory_frame = ttk.Frame(self)
        self.trajectory_frame.grid(row=0, column=0, sticky="nsew")
        self.scatter_particles = None  # Scatter plot for particles
        self.scatter_boundary = None   # Scatter plot for boundaries
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.frame = 0
        self.highlighted_particle_index = None
        self.highlighted_boundary_index = None

        # Right side for information graph
        self.graph_frame = ttk.Frame(self)
        self.graph_frame.grid(row=0, column=1, sticky="nsew")
        # Create a frame for the plot area within graph_frame
        self.plot_container = ttk.Frame(self.graph_frame)
        self.plot_container.pack(expand=True, fill=tk.BOTH)
        # ptl or bnd choose
        self.choice_label = ttk.Label(self.graph_frame, text="Choose Data Type:")
        self.choice_label.pack()
        self.data_type = tk.StringVar(value="Particle")
        self.data_type_dropdown = ttk.Combobox(self.graph_frame, textvariable=self.data_type, values=["Particle", "Boundary"], state="readonly")
        self.data_type_dropdown.pack()
        # index
        self.index_label = ttk.Label(self.graph_frame, text="Enter Index:")
        self.index_label.pack()
        # input window
        self.index_entry = ttk.Entry(self.graph_frame)
        self.index_entry.pack()
        # enter data button
        self.show_button = ttk.Button(self.graph_frame, text="Show Data", command=self.plot_data)
        self.show_button.pack()

        self.init_plot()
        self.animate_trajectory()


    def init_plot(self):
        """Initialize the scatter plots and axis limits."""
        scale = max(self.x, self.y)
        self.ax.set_xlim([-0.05 * scale, scale + 0.05 * scale])
        self.ax.set_ylim([-0.05 * scale, scale + 0.05 * scale])
        # Initialize scatter plot with the first frame's data
        self.scatter_particles = self.ax.scatter(self.trajectory[0, :, 0], self.trajectory[0, :, 1], s=self.ptl_size, c="blue", label="Particles")
        self.scatter_boundary = self.ax.scatter(self.boundary[:, 0], self.boundary[:, 1], s=self.ptl_size, c="black", label="Boundary")
        self.ax.set_title("SPH Simulation - Particle Trajectory")
        self.ax.legend()
        # Embed the plot into tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.trajectory_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()


    def animate_trajectory(self):
        """Update the scatter plot with the next frame's data."""
        self.frame += int(1/(self.dt*100))  # Move to the next frame
        if self.frame >= len(self.trajectory):
            self.frame = 0  # Loop back to the first frame
        # Update particle positions
        self.scatter_particles.set_offsets(self.trajectory[self.frame, :, :2])
        # Update boundary positions
        self.scatter_boundary.set_offsets(self.boundary[:, :2])
        # Update particle colors
        self.update_particle_colors()
        # Redraw the canvas
        self.canvas.draw()
        # Update the vertical line in the right-side plot
        self.update_graph_line()
        # Schedule the next update
        self.after(10, self.animate_trajectory)  # Update every 10 ms (10 FPS)


    def highlight(self, index, data_type):
        """Highlight a specific particle or boundary in red."""
        if data_type == "Particle":
            self.highlighted_particle_index = index
            self.highlighted_boundary_index = None
        elif data_type == "Boundary":
            self.highlighted_particle_index = None
            self.highlighted_boundary_index = index
        self.update_particle_colors()


    def update_particle_colors(self):
        """Update the color of particles and boundaries based on the highlighted index."""
        if self.highlighted_particle_index is not None:
            # Update the color for the highlighted particle
            positions = self.trajectory[self.frame, :, :2]
            colors = np.array(['red' if i == self.highlighted_particle_index else 'blue'
                            for i in range(len(positions))])
            sizes = np.array([self.highlight_size if i == self.highlighted_particle_index else self.ptl_size
                          for i in range(len(positions))])
            self.scatter_particles.set_edgecolor(colors)
            self.scatter_particles.set_sizes(sizes)
        else:
            self.scatter_particles.set_edgecolor('blue')
        
        if self.highlighted_boundary_index is not None:
            # Update the color for the highlighted boundary
            boundary_positions = self.boundary[:, :2]
            boundary_colors = np.array(['red' if i == self.highlighted_boundary_index else 'black'
                                        for i in range(len(boundary_positions))])
            boundary_sizes = np.array([self.highlight_size if i == self.highlighted_boundary_index else self.ptl_size
                          for i in range(len(boundary_positions))])
            self.scatter_boundary.set_edgecolor(boundary_colors)
            self.scatter_boundary.set_sizes(boundary_sizes)
        else:
            self.scatter_boundary.set_edgecolor('black')


    def plot_data(self):
        """Plot data for either particle or boundary based on user selection."""
        data_type = self.data_type.get()  # Either "Particle" or "Boundary"
        try:
            index = int(self.index_entry.get())
            time = self.dt*np.arange(self.ptl_rho_history.shape[0])
            self.highlight(index, data_type)

            for widget in self.plot_container.winfo_children():
                widget.destroy() 

            fig, axs = plt.subplots(4, figsize=(6, 7))
            if data_type == "Particle":
                # Plot rho history for particles
                axs[0].plot(time, self.ptl_rho_history[:, index, 0])
                axs[0].set_title(f'Particle {index} - Density (Rho)')
                axs[0].set_xlabel('Time')
                axs[0].set_ylabel('Density')
                # Plot pressure history for particles
                axs[1].plot(time, self.ptl_pres_history[:, index, 0])
                axs[1].set_title(f'Particle {index} - Pressure')
                axs[1].set_xlabel('Time')
                axs[1].set_ylabel('Pressure')
                # Plot pressure force components for particles
                axs[2].plot(time, self.ptl_pres_force_history[:, index, 0], label="Pressure a_x")
                axs[2].plot(time, self.ptl_pres_force_history[:, index, 1], label="Pressure a_y")
                axs[2].set_title(f'Particle {index} - Pressure Force Acceleration Components')
                axs[2].set_xlabel('Time')
                axs[2].set_ylabel('Acceleration')
                axs[2].legend()
                # Plot velocity components for particles
                axs[3].plot(time, self.trajectory[:, index, 2], label="v_x")
                axs[3].plot(time, self.trajectory[:, index, 3], label="v_y")
                axs[3].set_title(f'Particle {index} - Velocity Components')
                axs[3].set_xlabel('Time')
                axs[3].set_ylabel('Velocity')
                axs[3].legend()
            elif data_type == "Boundary":
                # Plot rho history for boundary particles
                axs[0].plot(time, self.bnd_rho_history[:, index, 0])
                axs[0].set_title(f'Boundary {index} - Density (Rho)')
                axs[0].set_xlabel('Time')
                axs[0].set_ylabel('Density')
                # Plot pressure history for boundary particles
                axs[1].plot(time, self.bnd_pres_history[:, index, 0])
                axs[1].set_title(f'Boundary {index} - Pressure')
                axs[1].set_xlabel('Time')
                axs[1].set_ylabel('Pressure')
            plt.subplots_adjust(hspace=1)
            canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
            canvas.draw()
            canvas.get_tk_widget().pack()
            self.right_fig = fig
            self.right_axs = axs
        except ValueError:
            print("Invalid input. Please enter a valid index.")
        except IndexError:
            print("Index out of range. Please try another index.")
        

    def update_graph_line(self):
        """Update the vertical line in the right-side plot to match the left-side animation time."""
        if hasattr(self, 'right_axs'):
            # Remove old vertical lines (if any)
            for ax in self.right_axs:
                for line in ax.lines:
                    if line.get_linestyle() == '--':  # Assuming the line to remove is dashed
                        line.remove()

            # Add new vertical lines at current frame time
            for ax in self.right_axs:
                ax.axvline(x=self.frame * self.dt, color='grey', linestyle='--')  # Scale frame time by time-step (0.01s)

            # Redraw the figure
            self.right_fig.canvas.draw()