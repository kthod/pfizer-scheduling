from ortools.sat.python import cp_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import time

import pandas as pd

# Load the Excel file
file_path = r"C:\Users\thodoris\Documents\Pfizer\result.xlsx"  # Replace with your Excel file path


# Specify the column of interest
column_name = 'MATERIAL_KEY'  # Replace with the column you want to extract

lines_list = [2,5,10, 20,50,100,200,500,800,1000,1500]
num_of_variables = []
num_of_constraints =  []
elapsed_time = []
#lines = 1000
# Extract the column values as a list
for lines in lines_list:
    df = pd.read_excel(file_path)
    p = df['TARGET_PROC_TIME'][:lines].tolist()
    Material_key = df['MATERIAL_KEY'][:lines].tolist()
    process_order = df['PROCESS_ORDER'][:lines].tolist()
    Jobs = df['MATERIAL_KEY'][:lines].tolist()
    Brands = df['BRAND_DESC'][:lines].tolist()
    mg = df['API1_STRENGTH_KEY'][:lines].tolist()
    machines = df['WORK_CENTER_RESOURCE_x'][:lines].tolist()
    alternative_machines = df['WORK_CENTER_RESOURCE_y'][:lines].tolist()
    # Print the resulting array

    Jobs = list(zip(Material_key,process_order))
  



    machines_set = set(machines + alternative_machines)  

    # Map each distinct element to a unique number
    machines_to_number = {machine: idx for idx, machine in enumerate(machines_set)}

    


    # Example data
    num_jobs = len(set(Jobs))
    num_tasks = len(p)
    num_machines = len(machines_set)

   
    alternative_machines_dict = {}
    for task in range(num_tasks):
        key = (Material_key[task], machines[task])  # Create a tuple key from A and B
        value = alternative_machines[task]
        alternative_machines_dict.setdefault(key, set()).add(machines_to_number[value])  # Add the value to the list for the key

    # Create a dictionary to store task-machine assignments
    task_machine_mapping = {}

    # Assign each task a random subset of machines it can be executed on
    for task in range( num_tasks ):
        #machines_for_task = random.sample(range( num_machines ), random.randint(1, num_machines-1))
        task_machine_mapping[task] = {machines_to_number[machines[task]]}
        task_machine_mapping[task].update(alternative_machines_dict[(Material_key[task], machines[task])])



    # Display the first few entries to verify the structure
    #import pandas as pd

  


    #Utilization rates
    u_r = num_machines*[1]#np.random.uniform(1, 1, num_machines)  # Machine efficiencies
    #D = np.random.uniform(100, 150, num_jobs).astype(int)  # Job deadlines
    M =1000

    #delta parameters that indicates whether task have the same brand
    delta = np.zeros((num_tasks,num_tasks))
    for i in range(num_tasks):
        for k in range(num_tasks):
            if Brands[i]!= Brands[k]:
                delta[i][k] = 1

    #delta star parameters that indicates whether task have the same API1 key strength
    delta_star = np.zeros((num_tasks,num_tasks))
    for i in range(num_tasks):
        for k in range(num_tasks):
            if mg[i]!= mg[k]:
                delta_star[i][k] = 1


    t_c = 10  #cleaning time for different brand
    t_c_star=5  #cleaning time for different api1 key

    scale = 1000
    # Define the model
    model = cp_model.CpModel()

    # Decision variables
    S = [model.NewIntVar(0, 1500, f'S_{i}') for i in range(num_tasks)]  # Start times

    #machine assignment variable
    X = [[None for m in range(num_machines)] for i in range(num_tasks)]
    for i in range(num_tasks):
        for m in range(num_machines):
            if m in task_machine_mapping[i]:
                X[i][m]=model.NewBoolVar(f'X_{i}_{m}')

    #sequencing variables
    Y = [[None for k in range(num_tasks)] for i in range(num_tasks)]
    for i in range(num_tasks):
        for k in range(num_tasks):
            if  set(task_machine_mapping[i]) & set(task_machine_mapping[k]):
                Y[i][k]=model.NewBoolVar(f'Y_{i}_{k}')

    #inactivity period sequencing variables
    Z = [[model.NewBoolVar(f'Z_{i}_{l}') for l in range(3)] for i in range(num_tasks)]  # Maintenance

    C_max = model.NewIntVar(0, 1500, 'C_max')  # Makespan

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
            if set(task_machine_mapping[i]) & set(task_machine_mapping[k]):
                model.Add(Y[i][k] + Y[k][i] <= 1)

    # Job Deadline Constraint
    # for j in range(num_jobs):
    #     for i in range(num_tasks):
    #         if Jobs[i] == j:
    #             for m in task_machine_mapping[i]:
    #                 model.Add(scale * (S[i] + int(p[i]* (2 - u_r[m]))) <= scale * D[j]).OnlyEnforceIf(X[i][m])

    # No Machine Overlap
    for i in range(num_tasks):
        for k in range(num_tasks):
            if set(task_machine_mapping[i]) & set(task_machine_mapping[k]):
                for m in (set(task_machine_mapping[i]) & set(task_machine_mapping[k])) :
                    model.Add(scale * S[k] >= scale * (S[i] + int(p[i] * (2 - u_r[m]) + int(delta[i][k]*t_c + (1-delta[i][k])*delta_star[i][k]*t_c_star)))).OnlyEnforceIf([Y[i][k], X[i][m], X[k][m]])
                    model.Add(scale * S[i] >= scale * (S[k] + int(p[k] * (2 - u_r[m]) + int(delta[i][k]*t_c + (1-delta[i][k])*delta_star[i][k]*t_c_star)))).OnlyEnforceIf([Y[k][i], X[i][m], X[k][m]])

    # Solve the model
    print(lines)
    proto = model.Proto()
    print(len(proto.variables))
    print(len(proto.constraints))
    
# Get the number of variables
    num_of_variables.append(len(proto.variables))
    num_of_constraints.append(len(proto.constraints))

    start_time = time.time()
    solver = cp_model.CpSolver()
    end_time = time.time()

    elapsed_time.append(end_time - start_time)
    status = solver.Solve(model)

    print(solver.Solve(model))


plt.plot(lines_list, num_of_variables)

plt.xlabel('number of tasks')
plt.ylabel('number of variables')
plt.title("Number of Variables")
plt.show()


plt.plot(lines_list, num_of_constraints)


plt.xlabel('number of tasks')
plt.ylabel('number of constraints')
plt.title("Number of Constraints")
plt.show()

plt.plot(lines_list, elapsed_time)


plt.xlabel('number of tasks')
plt.ylabel('time')
plt.title("Elapsed time")
plt.show()
# Check and plot results
# if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
#     print("Solution Found:")
#     tasks_schedule = []
#     for i in range(num_tasks):
#         start_time = solver.Value(S[i])
#         duration = p[i]
#         assigned_machine = None
#         for m in task_machine_mapping[i]:
#             if solver.Value(X[i][m]) == 1:
#                 assigned_machine = f'Machine {m }'
#                 duration = p[i] * (2 - u_r[m])
#                 break
#         tasks_schedule.append((i, assigned_machine, start_time, duration))
#         print(f"Task {i}: Start = {start_time}, Duration = {duration}, Assigned Machine = {assigned_machine}")

#     # Plotting the Gantt chart
#     colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
#     job_labels = list(set(Jobs))
#     color_map = {job_labels[i]: colors[i % len(colors)] for i in range(len(job_labels))}

#     fig, ax = plt.subplots(figsize=(10, 6))
#     for task_id, machine, start_time, duration in tasks_schedule:
#         color = color_map[Jobs[task_id]]
#         ax.barh(machine, duration, left=start_time, color=color, edgecolor='black')
#         ax.text(start_time + duration / 2, machine, f'Task {task_id}', ha='center', va='center', color='white', fontsize=4)

#     ax.set_xlabel("Time")
#     ax.set_ylabel("Machines")
#     ax.set_title("Gantt Chart of Task Schedule")
#     ax.grid(True)

#     legend_handles = [mpatches.Patch(color=color_map[job], label=job) for job in job_labels]
#     ax.legend(handles=legend_handles, title="Jobs")
#     plt.show()
# else:
#     print("No feasible solution found.")
