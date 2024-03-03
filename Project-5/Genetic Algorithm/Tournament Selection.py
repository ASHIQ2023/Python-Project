import pandas as pd
import random 
import random
import pandas as pd

df_a = {'String':[],'Fitness':[]}
No_of_input = int(input("How many values do you want to enter:\n"))
for i in range(No_of_input):
    b = str(input("enter the string value:\n"))
    df_a['String'].append(b)
    c = int(input("enter the fitness value:\n"))
    df_a['Fitness'].append(c)

a = pd.DataFrame(df_a)
b = pd.DataFrame(a)

if len(b)%2 == 0:
    tournament_size = (len(b['String']))/2 + 1
else:
    tournament_size = len(b['String'])

tournament_size = int(tournament_size)    
parents = []

for i in range(tournament_size):
    tournament = random.sample(range(len(b)), 2) # will play the tournament randomly
    if b.loc[tournament[0], 'Fitness'] > b.loc[tournament[0], 'Fitness']:
        parents.append((b.loc[tournament[0], 'String']), b.loc([tournament[0],'Fitness']))
    else:
        parents.append((b.loc[tournament[1], 'String'], b.loc[tournament[1],'Fitness']))

chosen_string = parents 

df_chosen_string = pd.DataFrame(chosen_string, columns=['Chosen String','Fitness'])

df_final_results_tournament_selection = pd.concat([b, df_chosen_string], axis=1)

print(df_final_results_tournament_selection)