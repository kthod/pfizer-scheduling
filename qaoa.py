from qiskit import *
import dimod
from scipy.optimize import minimize
import numpy as np
from qiskit.visualization import plot_histogram

import matplotlib.pyplot as plt

class MonitoredQAOA:

    def __init__(self, qubo_matrix: list[list[float]], layers: int = 1) -> None:
        
        self.nq = len(qubo_matrix)
        self.qubo = qubo_matrix
        self.layers = layers
        self.cost_evolution = list()

        self.experiments = list()
        self.qubo_dict = dict()
        self.total_counts = dict()

        for i in range(self.nq):
            for j in range(i, self.nq):
                self.qubo_dict[(i, j)] = qubo_matrix[i][j]


        h, J, offset = dimod.qubo_to_ising(self.qubo_dict)

        self.h = h
        self.J = J

        self.normalization = np.max([
            np.max([np.abs(val) for val in self.h.values()]),
            np.max([np.abs(val) for val in self.J.values()])
        ])

        self.qubo / self.normalization

    def build_circuit(self, theta) -> QuantumCircuit:

        qc = QuantumCircuit(self.nq)

        qc.h(list(range(self.nq)))

        p = len(theta) // 2

        gamma = theta[:p]
        beta = theta[p:]

        for iter in range(self.layers):
            
            for key in self.J.keys():

                if self.J[key] == 0:
                    continue

                i = key[0]
                j = key[1]

                qc.rzz(gamma[iter] * self.J[key] / self.normalization, i, j)
            
            qc.barrier()

            for key in self.h.keys():

                if self.h[key] == 0:
                    continue

                qc.rz(gamma[iter] * self.h[key] / self.normalization, key)

            qc.barrier()

            for i in range(self.nq):
                qc.rx(beta[iter], i)

        qc.measure_all()

        return qc
    
    def compute_cost(self, key) -> float:

        bitstring = key[::-1]

        x = np.zeros(self.nq)
        for i in range(self.nq):
            x[i] = 1 if bitstring[i] == '1' else 0

        return (x.T @ self.qubo @ x)


    def cost_function(self, counts: dict) -> float:
        
        cost = 0.0

        for key in counts.keys():

            bitstring = key[::-1]

            x = np.zeros(self.nq)

            for i in range(self.nq):
                x[i] = 1 if bitstring[i] == '1' else 0
            
            cost += (counts[key] * (x.T @ self.qubo @ x)) 

        cost /= 1000

        self.cost_evolution.append(cost+16)

        return cost


    def get_expectation(self):

        backend = Aer.get_backend('qasm_simulator')

        def objective(theta) -> float:

            qc = self.build_circuit(theta)
            counts = execute(qc, backend, shots=1000).result().get_counts()
            return self.cost_function(counts)
        
        return objective
    
    def optimize(self, experiments: int = 1, maxiter: int = 1000):

        minimum_value = np.inf
        best_answer = None

        for _ in range(experiments):

            self.cost_evolution = list()

            obj = self.get_expectation()

            theta = [0.0] * (2 * self.layers)

            result = minimize(obj, theta, method="COBYLA", options={"maxiter":maxiter})

            self.experiments.append(np.array(self.cost_evolution  + ([self.cost_evolution[-1]] * (maxiter - len(self.cost_evolution)))))

            # Execute the circuit one last time...

            qc_final = self.build_circuit(result.x)

            counts_final = execute(qc_final, backend=Aer.get_backend('qasm_simulator'), shots=10000).result().get_counts()

            for key in counts_final.keys():

                rev = key[::-1]

                if rev not in self.total_counts.keys():
                    self.total_counts[rev] = counts_final[key]
                else:
                    self.total_counts[rev] += counts_final[key]

                ccc = self.compute_cost(key)

                if ccc < minimum_value:
                    minimum_value = ccc
                    best_answer = key[::-1]

        self.experiments = np.array(self.experiments)

       

        
        upper = [np.max(self.experiments[:, i]) for i in range(maxiter)]
        lower = [np.min(self.experiments[:, i]) for i in range(maxiter)]
        mean = [np.mean(self.experiments[:, i]) for i in range(maxiter)]

        plt.figure()
        plt.fill_between(range(maxiter), upper, lower, color='lightblue')
        plt.plot(range(maxiter), mean, color='blue', linestyle='--',label = "QAOA")
        # plt.xlabel('Optimization Iteration')
        # plt.ylabel('Cost')
        # plt.title('Cost Evolution')
        # plt.show()

        # plot_histogram(self.total_counts)
        # plt.show()
        
        return best_answer



if __name__ == "__main__":

    
    a = [1,1,1,1]
    S=4
    def get_qubomat(a,S):
    # Adjacency is essentially a matrix which tells you which nodes are connected.
        matrix = np.zeros((len(a),len(a)))
        nvar = len(matrix)

        for i in range(len(a)):
            for j in range(i,len(a)):
                if i == j:
                    matrix[i][i] = (a[i]**2) - 2*S*a[i]
                else:
                    matrix[i][j] = 2*a[i]*a[j]
        return matrix
    # inst = Quantum_MPC(epsilon=epsilon, de=de, C=C, Horizon=Horizon, DT=DT, layers=2)
    Q= get_qubomat(a,S)

    solver = MonitoredQAOA(qubo_matrix=Q, layers=1)

    solver.optimize(experiments=5, maxiter=200)