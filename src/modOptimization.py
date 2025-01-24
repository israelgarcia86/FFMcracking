# -*- coding: utf-8 -*-
"""Module containing all the definitions necessary to precit a crack onset using FFM
Created on Thu Jan 23 18:18:37 2025
@author: israelgarcia86
HOW TO USE: Please see Readme in the same repository
"""

import math
import random
import numpy as np
# import matplotlib.pyplot as plt

# # Define the functions
# def f1(x):
#     return math.sin(3 * x) + x**2 - 0.7 * x

# def f2(x):
#     return math.cos(2 * x) - x**2 + 0.5

# # Objective: Minimize the min of f1 and f2
# def objective(x):
#     return max(f1(x), f2(x))
    

# Gaussian process kernel (RBF)
def rbf_kernel(x1, x2, length_scale=1.0):
    return math.exp(-((x1 - x2)**2) / (2 * length_scale**2))

# Build the surrogate model (Gaussian process)
def gaussian_process(x_train, y_train, x_pred, noise=1e-6):
    n = len(x_train)
    K = np.array([[rbf_kernel(xi, xj) for xj in x_train] for xi in x_train])
    K += noise * np.eye(n)
    K_inv = np.linalg.inv(K)
    
    k_star = np.array([rbf_kernel(x_pred, xi) for xi in x_train])
    k_star_star = rbf_kernel(x_pred, x_pred)
    
    mean = np.dot(k_star,np.dot(K_inv,y_train))
    variance = k_star_star - np.dot(k_star,np.dot(K_inv,k_star))
    return mean, max(0, variance)

# Expected Improvement (EI) acquisition function
def expected_improvement(x_pred, x_train, y_train, best_y, xi=0.01):
    mean, variance = gaussian_process(x_train, y_train, x_pred)
    std_dev = math.sqrt(variance)
    z = (best_y - mean - xi) / std_dev if std_dev > 0 else 0
    ei = (best_y - mean - xi) * norm_cdf(z) + std_dev * norm_pdf(z)
    return ei

# Normal distribution CDF and PDF
def norm_pdf(z):
    return math.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)

def norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

# Bayesian optimization
def bayesian_optimization(obj_func, bounds, n_iterations, n_initial=5):
    x_train = [random.uniform(*bounds) for _ in range(n_initial)]
    y_train = [obj_func(x) for x in x_train]
    best_x = x_train[np.argmin(y_train)]
    best_y = min(y_train)
    
    for _ in range(n_iterations):
        x_candidates = np.linspace(bounds[0], bounds[1], 100)
        ei_values = [expected_improvement(x, x_train, y_train, best_y) for x in x_candidates]
        next_x = x_candidates[np.argmax(ei_values)]
        
        next_y = obj_func(next_x)
        x_train.append(next_x)
        y_train.append(next_y)
        
        if next_y < best_y:
            best_x, best_y = next_x, next_y
    
    return best_x, best_y, x_train, y_train

# # Run the optimization
# bounds = (0, 2)
# n_iterations = 10
# best_x, best_y, x_train, y_train = bayesian_optimization(objective, bounds, n_iterations)

# print('Best x: ' + str(best_x))
# print('Best y: ' + str(best_y))

# # Plot the functions and the result
# x_vals = np.linspace(bounds[0], bounds[1], 500)
# f1_vals = [f1(x) for x in x_vals]
# f2_vals = [f2(x) for x in x_vals]
# objective_vals = [objective(x) for x in x_vals]

# plt.figure(figsize=(10, 6))
# plt.plot(x_vals, f1_vals, label="f1(x)", linestyle="--")
# plt.plot(x_vals, f2_vals, label="f2(x)", linestyle="--")
# plt.plot(x_vals, objective_vals, label="max(f1(x), f2(x))", linewidth=2)
# plt.scatter(x_train, y_train, color="red", label="Sampled Points")
# plt.scatter([best_x], [best_y], color="green", s=100, label="Best Point")
# plt.title("Bayesian Optimization of min(f1(x), f2(x))")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.grid(True)
# plt.show()
