import random
import pandas as pd
import numpy as np

# Helper Funtion
def single_point_crossover(parent1, parent2):
    assert len(parent1) == len(parent2) # ensure parents have the same length
    crossover_point = random.randint(0, len(parent1)-1)# select a random crossover point
    child1 = parent1[:crossover_point] + parent2[crossover_point:] # perform crossover
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    
    return child1, child2

def two_point_crossover(parent1, parent2):
    assert len(parent1) == len(parent2)
    crossover_points = sorted(random.sample(range(1, len(parent1)), 2))
    child1 = parent1[:crossover_points[0]] + parent2[crossover_points[0]:crossover_points[1]] + parent1[crossover_points[1]:]
    child2 = parent2[:crossover_points[0]] + parent1[crossover_points[0]:crossover_points[1]] + parent2[crossover_points[1]:] 
    return child1, child2

def uniform_crossover(parent1, parent2):
    assert len(parent1) == len(parent2)
    
    child1 = []
    child2 = []
    
    for bit1, bit2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(bit1)
            child2.append(bit2)
        else:
            child2.append(bit1)
            child1.append(bit2)
        
        return''.join(child1), ''.join(child2)


# Main Crossover Function
if __name__ == "__main__":  
    
    parent1 = str(input("Enter the parent1 binary code:\n"))
    parent2 = str(input("Enter the parent2 binary code:\n"))
    
        # perform the single point crossover
    child1, child2 = single_point_crossover(parent1, parent2)
    
    data = {'parent1':[parent1], 'parent2':[parent2], 'child1':[child1], 'child2':[child2]}
    single_point_crossover = pd.DataFrame(data)
    print(single_point_crossover)
    
    
    # perform multipoint crossover
    
    child1, child2 = two_point_crossover(parent1, parent2)
    
    data = {'parent1':[parent1], 'parent2':[parent2], 'child1':[child1], 'child2':[child2]}
    two_point_crossover = pd.DataFrame(data)
    print(two_point_crossover)
    
    
    
    # perform uniform crossover
    
    child1, child2 = uniform_crossover(parent1, parent2)
    
    data = {'parent1':[parent1], 'parent2':[parent2], 'child1':[child1], 'child2':[child2]}
    uniform_crossover = pd.DataFrame(data)
    print(uniform_crossover)