from ortools.linear_solver import pywraplp
import numpy as np
# Define the solver
solver = pywraplp.Solver.CreateSolver('GLOP')  

# Example data 
num_jobs = 5
num_tasks = 35  # 
num_machines = 4  # 
# Jobs = [0,0,0,0,0,
#         1,1,1,1,1,
#         2,2,2,2,2]


def generate_bounded_normal_array(size, lower_bound, upper_bound, mean=None, std_dev=None):
    """
    Generates an array of normally distributed random integers within specified bounds.

    Parameters:
    - size (int): Number of elements in the array.
    - lower_bound (int): Minimum value of the array elements.
    - upper_bound (int): Maximum value of the array elements.
    - mean (float): Mean of the normal distribution. Defaults to the midpoint of the bounds.
    - std_dev (float): Standard deviation of the normal distribution. Defaults to (upper_bound - lower_bound) / 4.

    Returns:
    - numpy.ndarray: Array of integers within the specified bounds.
    """
    # Set default mean and standard deviation if not provided
    if mean is None:
        mean = (lower_bound + upper_bound) / 2
    if std_dev is None:
        std_dev = (upper_bound - lower_bound) / 4  # A reasonable spread within the bounds

    # Generate normally distributed values
    random_array = np.random.normal(loc=mean, scale=std_dev, size=size)

    # Clip values to ensure they are within the specified range
    return np.clip(random_array, lower_bound, upper_bound)

    # Round to the nearest integer to get discrete values
    return np.round(random_array).astype(int)


def generate_uniform_array(size, num_of_options):
    """
    Generates an array of uniformly distributed integers from 0 to num_of_options - 1.

    Parameters:
    - size (int): Number of elements in the array.
    - num_of_options (int): The number of options (values from 0 to num_of_options - 1).

    Returns:
    - numpy.ndarray: Array of integers with uniform distribution in the range [0, num_of_options - 1].
    """
    return np.random.randint(0, num_of_options, size=size)



p = generate_bounded_normal_array(num_tasks, 1,10)

# Round to the nearest integer to get discrete values
p = np.round(p).astype(int)

Jobs = generate_uniform_array(num_tasks,num_jobs)

Brands = generate_uniform_array(num_jobs, 3)

mg = generate_uniform_array(num_tasks,3)
print("Brands")
print(Brands)

print("mg")
print(mg)

delta = np.zeros((num_tasks,num_tasks))
for i in range(num_tasks):
    for k in range(num_tasks):
        if Brands[Jobs[i]]!= Brands[Jobs[k]]:
            delta[i][k] = 1

delta_star = np.zeros((num_tasks,num_tasks))
for i in range(num_tasks):
    for k in range(num_tasks):
        if mg[i]!= mg[k]:
            delta_star[i][k] = 1

u_r = generate_bounded_normal_array(num_machines,0.8,1)

D = generate_bounded_normal_array(num_jobs,120,150)
# p = [2, 4, 6, 8, 10,
#      2, 3, 8, 6, 10,
#      5, 10, 5, 2, 8]  # Processing times for each task

M = 10000  # A large constant for big-M constraints
#u_r = [0.8, 0.9, 0.7]  # An example value for u_rm, modify as needed
scheduling_horizon = 150  # Define the scheduling horizon
#D = [100, 100, 100]  # Need by dates for each task, adjust as needed

# Extra parameters for overlapping / campaigning
t_c = 5  # Base campaign time, adjust as needed
t_c_star = 10  # Alternate campaign time, adjust as needed
#delta = [[1 if i != k else 0 for k in range(num_tasks)] for i in range(num_tasks)]  # Delta indicators
Y = [[solver.BoolVar(f'Y_{i}_{k}') for k in range(num_tasks)] for i in range(num_tasks)]

# Maintenance / Holiday parameters
num_windows = 3
w = [(0,0), (0,0), (50,60)]  # Maintenance start time
w_e = 30  # Maintenance end time
Z = [[solver.BoolVar(f'Z_{i}_{l}') for l in range(num_windows)] for i in range(num_tasks)]

# Decision Variables
S = [solver.IntVar(0, scheduling_horizon, f'S_{i}') for i in range(num_tasks)]
X = [[solver.BoolVar(f'X_{i}_{m}') for m in range(num_machines)] for i in range(num_tasks)]
C_max = solver.IntVar(0, scheduling_horizon, 'C_max')

# Objective Function: Minimize C_max
solver.Minimize(C_max)

# Constraints
# Completion Time Constraint
for i in range(num_tasks):
        for m in range(num_machines):
            solver.Add(C_max >= S[i] + p[i] + p[i] * (1 - u_r[m]) - M * (1 - X[i][m]))

# Task Order Constraint
for i in range(num_tasks):

        for k in range(i+1,num_tasks):
            if Jobs[i] == Jobs[k]:
                for m in range(num_machines):
                    solver.Add(S[k] >= S[i] + p[i] * (1 - u_r[m]) + p[i] - M * (1 - X[i][m]))

# Single Machine Assignment
for i in range(num_tasks):
    solver.Add(sum(X[i][m] for m in range(num_machines)) == 1)

for i in range(num_tasks):
    for k in range(num_tasks):
        if i != k:
            solver.Add(Y[i][k] + Y[k][i] == 1)

# # Need by Date Constraint
for j in range(num_jobs):
    for i in range(num_tasks):
        for m in range(num_machines):
            solver.Add(S[i] +  p[i] * (1 - u_r[m]) + p[i] - M * (1 - X[i][m]) <= D[Jobs[j]])

# No Machine Overlapping / Campaigning
for i in range(num_tasks):
    for k in range(num_tasks):
        if i != k:
            for m in range(num_machines):
                solver.Add(S[k] >= S[i] + p[i] * (2 - u_r[m]) + delta[i][k]*t_c + (1-delta[i][k])*delta_star[i][k]*t_c_star
                           - M * (1 - Y[i][k]) - M * (2 - X[i][m] - X[k][m]))
                solver.Add(S[i] >= S[k] + p[k] * (2 - u_r[m]) + delta[i][k]*t_c + (1-delta[i][k])*delta_star[i][k]*t_c_star
                           - M * Y[i][k] - M * (2 - X[i][m] - X[k][m]))

# Maintenance / Holidays Constraints
for i in range(num_tasks):
    for m in range(num_machines):
        for l in range(num_windows):
           # print(f"{w[l][0]}  {w[l][1]}")
            solver.Add(S[i] + p[i] * (2 - u_r[m]) <= w[l][0] + M * (1 - X[i][m]) + M * (1 - Z[i][l]))
            solver.Add(S[i] >= w[l][1] - M * (1 - X[i][m]) - M * Z[i][l])

# Solve the model
status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
    print("Solution Found:")
    tasks_schedule = []
    for i in range(num_tasks):
        start_time = S[i].solution_value()  # Use solution_value() here
        duration = p[i]
        assigned_machine = None
        for m in range(num_machines):
            if X[i][m].solution_value() == 1:  # Use solution_value() here
                assigned_machine = f'Machine {m + 1}'
                duration = p[i]* (2 - u_r[m]) 
                break
        tasks_schedule.append((i, assigned_machine, start_time, duration))
        print(f'Task {i}: Start = {start_time}, Duration = {duration}, Assigned Machine = {assigned_machine}')

    # for i in range(num_tasks):
    #     for k in range(num_tasks):
    #         print(f"{i} {k} {Y[i][k].solution_value()}")  # Use solution_value() here
    
    # for i in range(num_tasks):
    #     for l in range(num_windows):
    #         print(f"{i} {l} {Z[i][l].solution_value()}")  # Use solution_value() here
                
else:
    for i in range(num_tasks):
        for k in range(num_tasks):
            print(f"{i} {k} {Y[i][k].solution_value()}")  # Use solution_value() here
    
    for i in range(num_tasks):
        for l in range(num_windows):
            print(f"{i} {l} {Z[i][l].solution_value()}")  # Use solution_value() here
    print("No feasible solution found.")
    exit()


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
job_labels = list(set(Job for Job in Jobs))  # Unique machines
color_map = {job_labels[i]: colors[i % len(colors)] for i in range(len(job_labels))}

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each task as a bar in the Gantt chart
for task_id, machine, start_time, duration in tasks_schedule:
    color = color_map[Jobs[task_id]]
    ax.barh(machine, duration, left=start_time, color=color, edgecolor='black')
    ax.text(start_time + duration / 2, machine, f'Task {task_id}', ha='center', va='center', color='white')

# Customize the Gantt chart
ax.set_xlabel("Time")
ax.set_ylabel("Machines")
ax.set_title("Gantt Chart of Task Schedule")
ax.grid(True)

# Create a legend
legend_handles = [mpatches.Patch(color=color_map[job], label=job) for job in job_labels]
ax.legend(handles=legend_handles, title="Jobs")

# Show the Gantt chart
plt.show()