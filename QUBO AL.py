
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
import math
# Define epsilon and de arrays with specific values, representing parameters for the Quantum_MPC
# Create a random graph for the Max-Cut problem


# Example data
num_jobs = 2
num_tasks = 4
num_machines = 2
Horizon = 4
n = math.ceil(np.log2(Horizon))
p = [2,1,1,1]
Jobs = [0,0,1,1]
Brands = [0,0,1,0]
mg = [0,0,0,0]


# Create a dictionary to store task-machine assignments
task_machine_mapping = {}


task_machine_mapping[0] = [0]
task_machine_mapping[1] = [1]
task_machine_mapping[2] = [1]
task_machine_mapping[3] = [0]


u_r = num_machines*[1]#np.random.uniform(1, 1, num_machines)  # Machine efficiencies
D = np.random.uniform(100, 150, num_jobs).astype(int)  # Job deadlines
M =Horizon
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

t_c = 1
t_c_star = 0



tuples = [(0,3),(2,1)]
unique_mapping = {t: i for i, t in enumerate(set(tuples))}


def create_qubo_matrix(lambdas :list):
    Q = np.zeros((16,16))

    def objective_function():
        Q = np.zeros((16,16))
        Q[0][0] = 1
        Q[1][1] = 2
        return Q

    def last_task_quad(lambdas):

        Q = np.zeros((16,16))

        i = 2

        for j in range(num_tasks):
            len_machines = len(task_machine_mapping[j])
            m=1
            for machine in task_machine_mapping[j]:
                for q in range(n):
                    Q[q][q] += (2**q) * (2**q  -2 * p[j]*(2 - u_r[machine]) + 2 * M)*lambdas[j+m-1]

                for q in range(n-1):
                    for r in range(q+1, n):
                        Q[q][r] +=  2*(2**q)*(2**r) * lambdas[j+m-1]
                       ## Q[r][r] += (2**r) * (2**r  -2 * p[j]*(2 - u_r[machine]) + 2 * M)*lambdas[j+m-1]

                for q in range(n):
                    Q[i+q][i+q] += (2**q) * (2**q  +2 * p[j]*(2 - u_r[machine]) - 2 * M) * lambdas[j+m-1]

                for q in range(n-1):
                    for r in range(q+1, n):
                        Q[i+q][i+r] +=  2 * (2**q) * (2**r) * lambdas[j+m-1]
                       # Q[i+r][i+r] += (2**r) * (2**r  +2 * p[j]*(2 - u_r[machine]) + 2 * M)*lambdas[j+m-1]
   
                for q in range(n):
                    for r in range(n):
                        Q[q][i+r] += -2 * (2**q)*(2**r) * lambdas[j+m-1]


                Q[i+1+m][i+1+m] += (-M**2 + p[j]*(2 - u_r[machine])*M) * lambdas[j+m-1]

                for q in range(n):
                    Q[q][i+m+1] += -2 * (2**q) * M * lambdas[j+m-1]

                for q in range(n):
                    Q[i+q][i+m+1] += 2 * (2**q) * M * lambdas[j+m-1]


                m+=1
            
            i=i+n+len_machines
        
        return Q
    
    def last_task(lambdas: list):

        Q = np.zeros((16,16))

        i = 2

        for j in range(num_tasks):
            len_machines = len(task_machine_mapping[j])
            m=1
            for machine in task_machine_mapping[j+m-1]:

                for q in range(n):
                    Q[q][q] += -(2**q) * lambdas[j+m-1]

                for q in range(n):
                    Q[i+q][i+q] += (2**q) * lambdas[j+m-1]



                Q[i+1+m][i+1+m] += M*lambdas[j+m-1] 
                m+=1
            
            i=i+n+len_machines
        
        return Q
    # def last_task_quad(lambdas):

    #     Q = np.zeros((16,16))

    #     i = 2

    #     for j in range(num_tasks):
    #         len_machines = len(task_machine_mapping[j])
    #         m=1
    #         for machine in task_machine_mapping[j]:
    #             Q[0][0] += (1  -2 * p[j]*(2 - u_r[machine]) + 2 * M)*lambdas[j+m-1]
    #             Q[0][1] += 4 * lambdas[j+m-1]
    #             Q[1][1] += (4 + 2 * (-2* p[j]*(2 - u_r[machine])  + 2 * M)) * lambdas[j+m-1]

    #             Q[i][i] += (1 + 2 * p[j]*(2 - u_r[machine]) - 2 * M) * lambdas[j+m-1]
    #             Q[i][i+1] += 4 * lambdas[j+m-1]
    #             Q[i+1][i+1] += (4 + 2 * (2 * p[j]*(2 - u_r[machine]) - 2 * M)) * lambdas[j+m-1]

    #             Q[0][i] += -2 * lambdas[j+m-1]
    #             Q[0][i+1] += -4 * lambdas[j+m-1]
    #             Q[1][i] += -4 * lambdas[j+m-1]
    #             Q[1][i+1] += -8 * lambdas[j+m-1]

    #             Q[i+1+m][i+1+m] += (-M**2 + p[j]*(2 - u_r[machine])*M) * lambdas[j+m-1]

    #             Q[0][i+1+m] += -2 * M * lambdas[j+m-1]
    #             Q[1][i+1+m] += -4 * M * lambdas[j+m-1]

    #             Q[i][i+1+m] += 2 * M * lambdas[j+m-1]
    #             Q[i+1][i+1+m] += 4 * M * lambdas[j+m-1]

    #             m+=1
            
    #         i=i+n+len_machines
        
    #     return Q
    
    # def last_task(lambdas: list):

    #     Q = np.zeros((16,16))

    #     i = 2

    #     for j in range(num_tasks):
    #         len_machines = len(task_machine_mapping[j])
    #         m=1
    #         for machine in task_machine_mapping[j+m-1]:
    #             Q[0][0] += -1*lambdas[j+m-1]  
    #             Q[1][1] += -2*lambdas[j+m-1] 

    #             Q[i][i] += 1 *lambdas[j+m-1] 
    #             Q[i+1][i+1] += 2*lambdas[j+m-1] 

    #             Q[i+1+m][i+1+m] += M*lambdas[j+m-1] 
    #             m+=1
            
    #         i=i+n+len_machines
        
    #     return Q

    def task_order_quad(lambdas):

        Q = np.zeros((16,16))

        i = 2
        k=5
        #j=0
        l = 0
        for j  in [0,2]:
            for m in range(1,len(task_machine_mapping[j])+1):
                Q[k][k] += (1  -2 * p[j]*(2 - u_r[m]) + 2 * M) * lambdas[l]
                Q[k][k+1] += 4 * lambdas[l]
                Q[k+1][k+1] += (4 + 2 * (-2* p[j]*(2 - u_r[m])  + 2 * M)) * lambdas[l]

                Q[i][i] +=( 1 + 2 * p[j]*(2 - u_r[m]) - 2 * M) * lambdas[l]
                Q[i][i+1] += 4 * lambdas[l]
                Q[i+1][i+1] += (4 + 2 * (2 * p[j]*(2 - u_r[m]) - 2 * M)) * lambdas[l]

                Q[i][k] += -2 * lambdas[l]
                Q[i][k+1] += -4 * lambdas[l]
                Q[i+1][k] += -4 * lambdas[l]
                Q[i+1][k+1] += -8 * lambdas[l]

                Q[i+1+m][i+1+m] += (-M**2 + p[j]*(2 - u_r[m])*M) * lambdas[l]

                Q[i+m+1][k] += -2 * M * lambdas[l]
                Q[i+1+m][k+1] += -4 * M * lambdas[l]

                Q[i][i+1+m] += 2 * M * lambdas[l]
                Q[i+1][i+1+m] += 4 * M * lambdas[l]
            
            i=8
            k=11
            l+=1
        return Q
    
    def task_order(lambdas):



        Q = np.zeros((16,16))

        i = 2
        k=5
        j=0
        for j  in [0,1]:
            for m in range(1,len(task_machine_mapping[j])+1):
                Q[k][k] += -1*lambdas[j]  
                Q[k+1][k+1] += -2*lambdas[j] 

                Q[i][i] += 1 *lambdas[j] 
                Q[i+1][i+1] += 2*lambdas[j] 

                Q[i+1+m][i+1+m] += M*lambdas[j] 

            i=8
            k=11
        
        return Q
    

    def no_overlap_quad(lambdas):

        Q = np.zeros((16,16))

        i = 2
        k=11
        j=0
        l = 0
        for tuple  in tuples:
            y = unique_mapping[tuple] +14
            i = 3*tuple[0] + 2
            k = 3*tuple[1] + 2
            task_i = tuple[0]
            task_k = tuple[1]
            for m in range(1,len(task_machine_mapping[j])+1):
                A = p[task_i]*(2 - u_r[m]) +delta[tuple[0]][tuple[1]]*t_c +  (1-delta[tuple[0]][tuple[1]])*delta_star[tuple[0]][tuple[1]]*t_c_star
                B = p[task_k]*(2 - u_r[m]) +delta[tuple[0]][tuple[1]]*t_c +  (1-delta[tuple[0]][tuple[1]])*delta_star[tuple[0]][tuple[1]]*t_c_star
                #First Constraint
                Q[k][k] += (1 - 2 * A + 6 * M) * lambdas[l]
                Q[k][k+1] += 4 * lambdas[l]
                Q[k+1][k+1] += (4 + 2 * (- 2 * A + 6 * M)) * lambdas[l]
                #Second Constraint
                Q[k][k] += (1 +2 * B - 4 * M) * lambdas[l+1]
                Q[k][k+1] += 4 * lambdas[l+1]
                Q[k+1][k+1] += (4 + 2 * (2 * B - 4 * M)) * lambdas[l+1]

                #First Constraint
                Q[i][i] +=( 1 + 2 * A - 6 * M) * lambdas[l]
                Q[i][i+1] += 4 * lambdas[l]
                Q[i+1][i+1] += (4 + 2 * ( 2 * A - 6 * M)) * lambdas[l]
                #Second Constraint
                Q[i][i] +=( 1 - 2 * B + 4 * M) * lambdas[l+1]
                Q[i][i+1] += 4 * lambdas[l+1]
                Q[i+1][i+1] += (4 + 2 * ( -2 * B + 4 * M)) * lambdas[l+1]

                #First Constraint
                Q[i][k] += -2 * lambdas[l]
                Q[i][k+1] += -4 * lambdas[l]
                Q[i+1][k] += -4 * lambdas[l]
                Q[i+1][k+1] += -8 * lambdas[l]
                #Second Constrainta
                Q[i][k] += -2 * lambdas[l+1]
                Q[i][k+1] += -4 * lambdas[l+1]
                Q[i+1][k] += -4 * lambdas[l+1]
                Q[i+1][k+1] += -8 * lambdas[l+1]

                #First Constraint
                Q[i+1+m][i+1+m] += (-5*M**2 + 2 * M * A) * lambdas[l]
                Q[k+1+m][k+1+m] += (-5*M**2 + 2 * M * A) * lambdas[l]
                Q[y][y] += (-5*M**2 + 2 * M * A) * lambdas[l]
                #Second Constraint
                Q[i+1+m][i+1+m] += (-3*M**2 + 2 * M * B) * lambdas[l+1]
                Q[k+1+m][k+1+m] += (-3*M**2 + 2 * M * B) * lambdas[l+1]
                Q[y][y] += (5*M**2 - 2 * M * B) * lambdas[l+1]

                #First Constraint
                Q[i+m+1][k] += -2 * M * lambdas[l]
                Q[i+1+m][k+1] += -4 * M * lambdas[l]
                #Second Constraint
                Q[i+m+1][k] += 2 * M * lambdas[l+1]
                Q[i+1+m][k+1] += 4 * M * lambdas[l+1]

                #First Constraint 
                Q[i][i+1+m] += 2 * M * lambdas[l]
                Q[i+1][i+1+m] += 4 * M * lambdas[l]
                #Second Constrant
                Q[i][i+1+m] += -2 * M * lambdas[l+1]
                Q[i+1][i+1+m] += -4 * M * lambdas[l+1]

                #First Constraint
                Q[k+m+1][k] += -2 * M * lambdas[l]
                Q[k+1+m][k+1] += -4 * M * lambdas[l]
                #Second Constraint
                Q[k+m+1][k] += 2 * M * lambdas[l+1]
                Q[k+1+m][k+1] += 4 * M * lambdas[l+1]

                #First Constraint
                Q[i][k+1+m] += 2 * M * lambdas[l]
                Q[i+1][k+1+m] += 4 * M * lambdas[l]
                #Second Constraint
                Q[i][k+1+m] += -2 * M * lambdas[l+1]
                Q[i+1][k+1+m] += -4 * M * lambdas[l+1]

                #First Constraint
                Q[y][k] += -2 * M * lambdas[l]
                Q[y][k+1] += -4 * M * lambdas[l]
                #Second Constraint
                Q[y][k] += -2 * M * lambdas[l+1]
                Q[y][k+1] += -4 * M * lambdas[l+1]
                
                #First Constraint
                Q[i][y] += 2 * M * lambdas[l]
                Q[i+1][y] += 4 * M * lambdas[l]
                #Second Constraint
                Q[i][y] += 2 * M * lambdas[l+1]
                Q[i+1][y] += 4 * M * lambdas[l+1]

                #First Constraint
                Q[i+m+1][y] += 2 * M**2* lambdas[l]
                Q[k+m+1][y] += 2 * M**2* lambdas[l]
                Q[k+m+1][i+m+1] += 2 * M**2* lambdas[l]
                #Second Constraint
                Q[i+m+1][y] += -2 * M**2* lambdas[l]
                Q[k+m+1][y] += -2 * M**2* lambdas[l]
                Q[k+m+1][i+m+1] += 2 * M**2* lambdas[l]
            
            # i=8
            # k=11
                l+=2
        return Q
    
    def no_overlap(lambdas):



        Q = np.zeros((16,16))
        tuples = [(0,3),(2,1)]
        unique_mapping = {t: i for i, t in enumerate(set(tuples))}
        i = 2
        k=11
        j=0
        l = 0
        for tuple  in tuples:
            y = unique_mapping[tuple] +14
            i = 3*tuple[0] + 2
            k = 3*tuple[1] + 2
            j = tuple[0]
            for m in range(1,len(task_machine_mapping[j])+1):
                #First Constraint
                Q[k][k] += -1*lambdas[l]  
                Q[k+1][k+1] += -2*lambdas[l] 

                Q[i][i] += 1 *lambdas[l] 
                Q[i+1][i+1] += 2*lambdas[l] 

                Q[i+1+m][i+1+m] += M*lambdas[l] 
                Q[k+1+m][k+1+m] += M*lambdas[l] 
                Q[y][y] += M*lambdas[l] 

                #Second Constraint
                Q[k][k] += 1*lambdas[l+1]  
                Q[k+1][k+1] += 2*lambdas[l+1] 

                Q[i][i] += -1 *lambdas[l+1] 
                Q[i+1][i+1] += -2*lambdas[l+1] 

                Q[i+1+m][i+1+m] += -M*lambdas[l+1] 
                Q[k+1+m][k+1+m] += -M*lambdas[l+1] 
                Q[y][y] += -M*lambdas[l+1] 

                l+=2

            # i=8
            # k=11
        
        return Q
    # Adjacency is essentially a matrix which tells you which nodes are connected.

    Q = objective_function() + task_order(lambdas[0:2])+last_task(lambdas[2:6]) +0.5*task_order_quad(lambdas[0:2]) + 0.5*last_task_quad(lambdas[2:6]) +no_overlap(lambdas[6:]) + 0.5*no_overlap_quad(lambdas[6:])
        
        
    return  Q




# Augmented Lagrangian algorithm
def augmented_lagrangian_method(mu_init, lambda_init, rho, max_iterations=10, tol=1e-6):
    """
    Implements the Augmented Lagrangian method for inequality constraints.

    Args:
        mu_init (float): Initial penalty parameter (\mu > 0).
        lambda_init (np.array): Initial Lagrange multipliers (\lambda).
        rho (float): Increase factor for \mu (\rho > 1).
        max_iterations (int): Maximum number of iterations.
        tol (float): Tolerance for constraint satisfaction.

    Returns:
        lambda_final (np.array): Final Lagrange multipliers.
        mu_final (float): Final penalty parameter.
    """
    # Initialize parameters
    mu = mu_init
    rho =2
    lambdas = np.array(lambda_init)
    n_constraints = len(lambda_init)

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}:")
        
        # Example placeholder for x (binary variables)
        # Replace this with actual x obtained from solving the optimization problem

        # inst = Quantum_MPC(epsilon=epsilon, de=de, C=C, Horizon=Horizon, DT=DT, layers=2)
        Q= create_qubo_matrix(lambdas)
        number_of_experiments = 10

        print(Q)
        # solver = MonitoredQAOA(qubo_matrix=Q, layers=5)

        # solver.optimize(experiments=number_of_experiments, maxiter=150)


        inst_full = VQE(Q, layers=2,)
        inst_full.optimize(n_measurements=10000, number_of_experiments=number_of_experiments, maxiter=300)
        #print(inst_full.get_sched())  

        #inst_full.plot_evolution( color='C2')
        print(inst_full.show_solution())
        bitstring = inst_full.show_solution()
        # Set up the plot for evaluating the cost function over the optimization iterations
        # plt.title(f"Evaluation of Cost Function for timesteps", fontsize=16)
        # plt.ylabel('Cost Function', fontsize=14)
        # plt.xlabel('Iteration', fontsize=14)
        # plt.legend(fontsize=12)
        # plt.show()

        #print(f"  Current x: {x}")

        # Evaluate the constraint violations
        c_values = []
        start_time = []
        assigned_machine = []
        #print(f"  Current constraint violations: {c_values}")
        C_max = int(bitstring[0]) + 2*int(bitstring[1])
        for i in range(num_tasks):
            start_time.append(int(bitstring[2+i*3]) + 2*int(bitstring[2+i*3+1]))
            print(start_time)
            duration = p[i]
            for m in task_machine_mapping[i]:
                if int(bitstring[4+i*3]) == 1:
                    assigned_machine.append(m) 
        # Update Lagrange multipliers
        for i in range(2):
           # print(f"i {i} machine {assigned_machine[i]} ")
            c_values.append(-(start_time[2*i+1] - start_time[2*i] - p[2*i]))#*(2 - u_r[assigned_machine[i]])))
            if c_values[i] > 0:  # Only update if the constraint is violated
                lambdas[i] += mu * c_values[i]
                print(f"  Updated lambda[{i}]: {lambdas[i]}")

        for i in range(4):
           # print(f"i {i} machine {assigned_machine[i]} ")
            c_values.append(-(C_max - start_time[i] - p[i]))#*(2 - u_r[assigned_machine[i]])))
            if c_values[i+2] > 0:  # Only update if the constraint is violated
                lambdas[i+2] += mu * c_values[i+2]
                print(f"  Updated lambda[{i+2}]: {lambdas[i+2]}")

        tuples = [(0,3),(2,1)]
        for i in range(0,2):
            tuple = tuples[i]
            y_ind = unique_mapping[tuple] +14
           
            # print(f"{tuple} y {y}")
            y = int(bitstring[y_ind])
           # print(f"i {i} machine {assigned_machine[i]} ")
            c_values.append(-(start_time[tuple[1]] - start_time[tuple[0]] - p[tuple[0]]-delta[tuple[0]][tuple[1]]*t_c - (1-delta[tuple[0]][tuple[1]])*delta_star[tuple[0]][tuple[1]]*t_c_star + M*(1-y)))#*(2 - u_r[assigned_machine[i]])))
            if c_values[2*i+6] > 0:  # Only update if the constraint is violated
                lambdas[2*i+6] += mu * c_values[2*i+6]
                print(f"  Updated lambda[{2*i+6}]: {lambdas[2*i+6]}")
            
            c_values.append(-(start_time[tuple[0]] - start_time[tuple[1]] - p[tuple[1]]-delta[tuple[0]][tuple[1]]*t_c - (1-delta[tuple[0]][tuple[1]])*delta_star[tuple[0]][tuple[1]]*t_c_star + M*y))#*(2 - u_r[assigned_machine[i]])))
            if c_values[2*i+1+6] > 0:  # Only update if the constraint is violated
                lambdas[2*i+1+6] += mu * c_values[2*i+1+6]
                print(f"  Updated lambda[{2*i+1+6}]: {lambdas[2*i+1+6]}")

        # Check stopping criterion: all constraints satisfied
        if all(np.array(c_values) <= tol):
            print("Stopping criteria reached: all constraints satisfied.")
            break

        # Increase penalty parameter
        mu *= rho
        print(f"  Updated mu: {mu}")

    return lambdas, mu , bitstring

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
lambdas , mu , bitstring = augmented_lagrangian_method(mu_init=1,lambda_init=np.ones(10),rho=2)
tasks_schedule = []
for i in range(num_tasks):
    start_time = int(bitstring[2+i*3]) + 2*int(bitstring[2+i*3+1])
    print(start_time)
    duration = p[i]
    assigned_machine = 0
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


