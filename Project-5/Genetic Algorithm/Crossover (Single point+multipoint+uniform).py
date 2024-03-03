import random
import pandas as pd
import numpy as np

def single_point_crossover(parent1, parent2):
    assert len(parent1) == len(parent2) # ensure parents have the same length
    crossover_point = random.randint(0, len(parent1)-1)# select a random crossover point
    child1 = parent1[:crossover_point] + parent2[crossover_point:] # perform crossover
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    
    return child1, child2

if __name__ == "__main__":  
    
    parent1 = str(input("Enter the parent1 binary code:\n"))
    parent2 = str(input("Enter the parent2 binary code:\n"))
    
    # perform the single point crossover
    child1, child2 = single_point_crossover(parent1, parent2)
    
    data = {'parent1':[parent1], 'parent2':[parent2], 'child1':[child1], 'child2':[child2]}
    crossover = pd.DataFrame(data)
    print(crossover)
    
    
    # perform multipoint crossover
    
    
    # perform uniform crossover
    