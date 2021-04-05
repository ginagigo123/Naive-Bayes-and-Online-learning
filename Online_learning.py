# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:05:38 2021

@author: ginagigo
content:
    online learning 
"""
import numpy as np
import matplotlib.pyplot as plt

def combination(N, m):
    result = 1
    for i in range(m+1, N+1):
        result *= i
    for j in range(N-m, 1, -1):
        result /= j
    return result

def factorial(N):
    result = 1
    for i in range(2, N):
        result *= i
    return result

def prior(p, a, b):
    return p ** (a-1) * (1-p) ** (b-1) * factorial(a + b) / ( factorial(a) * factorial(b) )

def calLikelihood(N, m, p):
    return combination(N, m) * (p ** m) * ( (1-p) ** (N-m) )

def marginal(N, m, a, b, prob):
    sum = 0
    for p in prob:
        sum += p ** (m + a - 1) * ( 1 - p ) ** (N- m + b -1)
        #sum += prior(p, a, b) * calLikelihood(N, m, p)
    return sum
    

f = open('testfile.txt')

a,b = 0, 0
case = 1
for line in f:
    if line == "\n":
        continue
    
    line = line.strip('\n')
    print("case ", case, ": ", line.strip('\n'))
    i = 0
    count = [0, 0] # for 0, 1
    while i < len(line):
        count[int(line[i])] += 1
        i += 1
        
    p = count[0] / len(line)
    likelihood = calLikelihood(len(line), count[0], p)
    print("Likelihood:", likelihood)
    print("Beta prior    : ", a, b)
    a += count[1]
    b += count[0]
    print("Beta posterior: ", a, b)
    case += 1
    print()
    
f.close()

# Q3
f = open('testfile.txt')

a = int(input("a : "))
b = int(input("b : "))
case = 1
x_line = np.linspace(0, 1, 100)
for line in f:
    if line == "\n":
        continue
    
    line = line.strip('\n')
    i = 0
    count = [0, 0] # for 0, 1
    while i < len(line):
        count[int(line[i])] += 1
        i += 1
    
    m = count[0]
    N = len(line)
    
    # a != 0 & b!=0 -> draw figures:
    if a != 0 and b != 0:
        # fig
        fig, axes = plt.subplots(1, 3, figsize = (10,4))
        rows = 2
        columns = 2
        
        # calculate prior for the first time
        if case == 1:    
            y_prior_line = np.zeros(100)
            for index, x in enumerate(x_line):
                y_prior_line[index] = prior(x, a, b)
            
        axes[0].plot(x_line, y_prior_line, color="red")
        axes[0].set_title("prior")
        axes[0].set_xlabel("u")
        
         
        # show likelihood
        y_like_line = np.zeros(100)
        for index, x in enumerate(x_line):
            y_like_line[index] = calLikelihood(N, m, x)
        axes[1].plot(x_line, y_like_line, color="blue")
        axes[1].set_title("likelihood function")
        axes[1].set_xlabel("u")
        axes[1].set_ylim([0, 1])
        
        
        # calculate posterior
        y_marginal = marginal(len(line), count[0], a, b, x_line)
        y_post_line = np.zeros(100)
    
        for index, x in enumerate(x_line):
            y_post_line[index] = (x ** (m + a - 1) * (1 - x) ** (N - m + b - 1)) / y_marginal
        
        axes[2].plot(x_line, y_post_line, color="red")
        axes[2].set_title("posterior")
        axes[2].set_xlabel("u")
        
        # posterior = next prior
        y_prior_line = y_post_line
    
    p = count[0] / len(line)
    likelihood = calLikelihood(len(line), count[0], p)
    print("Likelihood:", likelihood)
    print("Beta prior    : ", a, b)
    a += count[1]
    b += count[0]
    print("Beta posterior: ", a, b)
    case += 1
    print()
    
f.close()

