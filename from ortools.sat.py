from ortools.sat.python import cp_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Example data
num_jobs = 3
num_tasks = 10
num_machines = 2

import random

# Number of tasks and machines

# Create a dictionary to store task-machine assignments
task_machine_mapping = {}

# Assign each task a random subset of machines it can be executed on
for task in range( num_tasks ):
    machines_for_task = random.sample(range( num_machines ), random.randint(1, num_machines-1))
    task_machine_mapping[task] = machines_for_task

# Display the first few entries to verify the structure
import pandas as pd

print(task_machine_mapping)


# Randomly generate the input data
p = np.round(np.random.uniform(1, 10, num_tasks)).astype(int)  # Task durations
Jobs = np.random.randint(0, num_jobs, num_tasks)  # Job assignments
Brands = np.random.randint(0, 3, num_jobs)  # Brand assignments
mg = np.random.randint(0, 3, num_tasks)  # mg assignments
u_r = num_machines*[1]#np.random.uniform(1, 1, num_machines)  # Machine efficiencies
D = np.random.uniform(100, 150, num_jobs).astype(int)  # Job deadlines
M =1000
delta = np.zeros((num_tasks,num_tasks))
for i in range(num_tasks):
    for k in range(num_tasks):
        if Brands[Jobs[i]]!= Brands[Jobs[k]]:
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

t_c = 10
t_c_star=5

print(Brands)

scale = 10
# Define the model
model = cp_model.CpModel()

# Decision variables
S = [model.NewIntVar(0, 150, f'S_{i}') for i in range(num_tasks)]  # Start times
X = [[None for m in range(num_machines)] for i in range(num_tasks)]
for i in range(num_tasks):
    for m in range(num_machines):
        if m in task_machine_mapping[i]:
            X[i][m]=model.NewBoolVar(f'X_{i}_{m}')
#X = [[[ if m in task_machine_mapping[i] for m in range(num_machines)] for i in range(num_tasks)]  # Task-machine assignments
Y = [[None for k in range(num_tasks)] for i in range(num_tasks)]
for i in range(num_tasks):
    for k in range(num_tasks):
        if  set(task_machine_mapping[i]) & set(task_machine_mapping[k]):
            Y[i][k]=model.NewBoolVar(f'Y_{i}_{k}')
#Y = [[model.NewBoolVar(f'Y_{i}_{k}') for k in range(num_tasks)] for i in range(num_tasks)]  # Task precedence
Z = [[model.NewBoolVar(f'Z_{i}_{l}') for l in range(3)] for i in range(num_tasks)]  # Maintenance

C_max = model.NewIntVar(0, 150, 'C_max')  # Makespan

# Objective: Minimize makespan
model.Minimize(C_max)

# Constraints
# Completion Time Constraint
for i in range(num_tasks):
    for m in task_machine_mapping[i]:
        model.Add(scale*C_max >= scale *  (S[i] +int( p[i] * (2 - u_r[m])))).OnlyEnforceIf(X[i][m])

# Task Order Constraint
for i in range(num_tasks):
    for k in range(i + 1, num_tasks):
        if Jobs[i] == Jobs[k]:
            for m in task_machine_mapping[i]:
                model.Add(scale * S[k] >= scale * (S[i] + int(p[i] * (2 - u_r[m])))).OnlyEnforceIf(X[i][m])

# Single Machine Assignment
for i in range(num_tasks):
    model.Add(sum(X[i][m] for m in task_machine_mapping[i]) == 1)

# Maintenance / Holidays Constraints
for i in range(num_tasks):
    for m in task_machine_mapping[i]:
        for l in range(3):
            start, end = 50, 60
            model.Add(scale * (S[i] + int(p[i] * (2 - u_r[m]))) <= scale * (start + M*(1-Z[i][l]))).OnlyEnforceIf(X[i][m])
            model.Add(scale * S[i] >= scale * (end- M*(Z[i][l]))).OnlyEnforceIf(X[i][m])

# Task Precedence
for i in range(num_tasks):
    for k in range(num_tasks):
        if i!=k:
            if set(task_machine_mapping[i]) & set(task_machine_mapping[k]):
                model.Add(Y[i][k] + Y[k][i] == 1)

# Job Deadline Constraint
for j in range(num_jobs):
    for i in range(num_tasks):
        if Jobs[i] == j:
            for m in task_machine_mapping[i]:
                model.Add(scale * (S[i] + int(p[i]* (2 - u_r[m]))) <= scale * D[j]).OnlyEnforceIf(X[i][m])

# No Machine Overlap
for i in range(num_tasks):
    for k in range(num_tasks):
        if set(task_machine_mapping[i]) & set(task_machine_mapping[k]):
            for m in (set(task_machine_mapping[i]) & set(task_machine_mapping[k])) :
                model.Add(scale * S[k] >= scale * (S[i] + int(p[i] * (2 - u_r[m]) + int(delta[i][k]*t_c + (1-delta[i][k])*delta_star[i][k]*t_c_star)))).OnlyEnforceIf([Y[i][k], X[i][m], X[k][m]])
                model.Add(scale * S[i] >= scale * (S[k] + int(p[k] * (2 - u_r[m]) + int(delta[i][k]*t_c + (1-delta[i][k])*delta_star[i][k]*t_c_star)))).OnlyEnforceIf([Y[k][i], X[i][m], X[k][m]])
print(delta)
print(delta_star)
# Solve the model
solver = cp_model.CpSolver()
status = solver.Solve(model)
# Check and plot results
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Solution Found:")
    tasks_schedule = []
    for i in range(num_tasks):
        start_time = solver.Value(S[i])
        duration = p[i]
        assigned_machine = None
        for m in task_machine_mapping[i]:
            if solver.Value(X[i][m]) == 1:
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
else:
    print("No feasible solution found.")
