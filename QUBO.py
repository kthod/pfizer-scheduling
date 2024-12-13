
# Import necessary libraries
from qiskit import *  # Import everything from Qiskit for quantum computing
import numpy as np  # Import NumPy for numerical operations
import networkx as nx  # Import NetworkX for graph operations (not used in the shown code)
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import matplotlib.colors as mcolors  # Import Matplotlib colors for color operations
from CompressedVQE_Class import CompressedVQE  # Import the Quantum_MPC class for quantum model predictive control
from VQE_Class import VQE 
from qaoa import MonitoredQAOA
import matplotlib.patches as mpatches
# Define epsilon and de arrays with specific values, representing parameters for the Quantum_MPC
# Create a random graph for the Max-Cut problem


# Example data
num_jobs = 2
num_tasks = 4
num_machines = 2

p = [2,1,2,1]
Jobs = [0,0,1,1]
Brands = [0,0,1,1]
mg = [0,0,0,0]


# Create a dictionary to store task-machine assignments
task_machine_mapping = {}


task_machine_mapping[0] = [0]
task_machine_mapping[1] = [0]
task_machine_mapping[2] = [1]
task_machine_mapping[3] = [1]


u_r = num_machines*[1]#np.random.uniform(1, 1, num_machines)  # Machine efficiencies
D = np.random.uniform(100, 150, num_jobs).astype(int)  # Job deadlines
M =10
delta = np.zeros((num_tasks,num_tasks))
for i in range(num_tasks):
    for k in range(num_tasks):
        if Brands[i]!= Brands[k]:
            delta[i][k] = 1
        else:
            delta[i][k] = 0

delta_star = np.zeros((num_tasks,num_tasks))
for i in range(num_tasks):
    for k in range(num_tasks):
        if mg[i]!= mg[k]:
            delta_star[i][k] = 1
        else:
            delta_star[i][k] = 0


def create_qubo_matrix():
    Q = np.zeros((14,14))

    def objective_function():
        Q = np.zeros((14,14))
        Q[0][0] = 1
        Q[1][1] = 2
        return Q

    def last_task():

        Q = np.zeros((14,14))

        i = 5
        j=1
        for i in [5,11]:
            for m in range(1,len(task_machine_mapping[j])+1):
                Q[0][0] += 1  -2 * p[j]*(2 - u_r[m]) + 2 * M
                Q[0][1] += 4
                Q[1][1] += 4 + 2 * (-2* p[j]*(2 - u_r[m])  + 2 * M)

                Q[i][i] += 1 + 2 * p[j]*(2 - u_r[m]) - 2 * M
                Q[i][i+1] += 4
                Q[i+1][i+1] += 4 + 2 * (2 * p[j]*(2 - u_r[m]) - 2 * M)

                Q[0][i] += -2
                Q[0][i+1] += -4
                Q[1][i] += -4
                Q[1][i+1] += -8

                Q[i+1+m][i+1+m] += -M**2 + p[j]*(2 - u_r[m])*M

                Q[0][i+1+m] += -2 * M
                Q[1][i+1+m] += -4 * M

                Q[i][i+1+m] += 2 * M
                Q[i+1][i+1+m] += 4 * M
            
            #i=i+m+2
        
        return Q

    def task_order():

        Q = np.zeros((14,14))

        i = 2
        k=5
        #j=0
        for j  in [0,2]:
            for m in range(1,len(task_machine_mapping[j])+1):
                Q[k][k] += 1  -2 * p[j]*(2 - u_r[m]) + 2 * M
                Q[k][k+1] += 4
                Q[k+1][k+1] += 4 + 2 * (-2* p[j]*(2 - u_r[m])  + 2 * M)

                Q[i][i] += 1 + 2 * p[j]*(2 - u_r[m]) - 2 * M
                Q[i][i+1] += 4
                Q[i+1][i+1] += 4 + 2 * (2 * p[j]*(2 - u_r[m]) - 2 * M)

                Q[i][k] += -2
                Q[i][k+1] += -4
                Q[i+1][k] += -4
                Q[i+1][k+1] += -8

                Q[i+1+m][i+1+m] += -M**2 + p[j]*(2 - u_r[m])*M

                Q[i+m+1][k] += -2 * M
                Q[i+1+m][k+1] += -4 * M

                Q[i][i+1+m] += 2 * M
                Q[i+1][i+1+m] += 4 * M
            
            i=8
            k=11
        
        return Q
    # Adjacency is essentially a matrix which tells you which nodes are connected.

    Q = objective_function() + 1*last_task() + task_order()
        
        
    return  Q
# inst = Quantum_MPC(epsilon=epsilon, de=de, C=C, Horizon=Horizon, DT=DT, layers=2)
Q= create_qubo_matrix()
number_of_experiments = 15
# # Optimize the Quantum MPC instance with specified parameters
# inst.optimize(n_measurements=20000, number_of_experiments=number_of_experiments, maxiter=300)
# Initialize an instance of Quantum_MPC with specific parameters for compressed VQE layer configuration
print(Q)
# solver = MonitoredQAOA(qubo_matrix=Q, layers=5)

# solver.optimize(experiments=number_of_experiments, maxiter=150)


# inst_min = CompressedVQE(Q, layers=2, na =1)

# # Optimize the Quantum MPC instance with specified parameters

# inst_min.optimize(n_measurements=10000, number_of_experiments=number_of_experiments, maxiter=150)
#print(inst_min.get_sched())  # Print the optimal schedule found by the optimization

#np.save('toy_example_sched_C1.npy', inst_min.get_sched())
# inst_min.plot_evolution()  # Plot the cost function evolution over iterations
# print(inst_min.optimal_params)
# print(inst_min.optimal_cost)
# print(inst_min.show_solution())
# Initialize another Quantum_MPC instance for a full encoding using the VQE algorithm
inst_full = VQE(Q, layers=2,)
inst_full.optimize(n_measurements=50000, number_of_experiments=number_of_experiments, maxiter=300)
#print(inst_full.get_sched())  

inst_full.plot_evolution( color='C2')
print(inst_full.show_solution())
# Set up the plot for evaluating the cost function over the optimization iterations
plt.title(f"Evaluation of Cost Function for timesteps", fontsize=16)
plt.ylabel('Cost Function', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.legend(fontsize=12)
plt.show()

# Plot the distribution of solutions for a given number of top solutions and shots
# scatter = inst_full.get_solution_distribution(normalization=[-S**2], solutions=10, shots=10000)
# #scatter = inst_min.get_solution_distribution(normalization=[-S**2], solutions=10, shots=10000)
# norm = mcolors.Normalize(vmin=0, vmax=1)
# plt.colorbar(mappable=scatter, norm=norm, label='Fraction of solutions')
# plt.yticks(range(1, number_of_experiments+1))
# plt.xlabel('Cost Function', fontsize=14)
# plt.ylabel('Optimization run', fontsize=14)
# plt.title(f"Distribution of solutions for  timesteps", fontsize=16)
# plt.grid(True)
# plt.legend(fontsize=12)
# plt.show()

print("Solution Found:")
bitstring = inst_full.show_solution()
tasks_schedule = []
for i in range(num_tasks):
    start_time = int(bitstring[2+i*3]) + 2*int(bitstring[2+i*3+1])
    print(start_time)
    duration = p[i]
    assigned_machine = None
    for m in task_machine_mapping[i]:
        if int(bitstring[4+i*3]) == 1:
            assigned_machine = f'Machine {m }'
            duration = p[i] * (2 - u_r[m])
            break
    tasks_schedule.append((i, assigned_machine, start_time, duration))
    print(f"Task {i}: Start = {start_time}, Duration = {duration}, Assigned Machine = {assigned_machine}, Job {Jobs[i]}, brand {Brands[Jobs[i]]}, mg {mg[i]}")

# Plotting the Gantt chart
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
job_labels = list(set(Jobs))
color_map = {job_labels[i]: colors[i % len(colors)] for i in range(len(job_labels))}

fig, ax = plt.subplots(figsize=(10, 6))
for task_id, machine, start_time, duration in tasks_schedule:
    color = color_map[Jobs[task_id]]
    ax.barh(machine, duration, left=start_time, color=color, edgecolor='black')
    ax.text(start_time + duration / 2, machine, f'Task {task_id}', ha='center', va='center', color='white', fontsize=6)

ax.set_xlabel("Time")
ax.set_ylabel("Machines")
ax.set_title("Gantt Chart of Task Schedule")
ax.grid(True)

legend_handles = [mpatches.Patch(color=color_map[job], label=job) for job in job_labels]
ax.legend(handles=legend_handles, title="Jobs")
plt.show()

