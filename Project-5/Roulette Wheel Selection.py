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

probability_value = []  # Initialize the probability_value list
sum_fitness = sum(b['Fitness'])

# Calculate the probability values
for i in range(len(b['Fitness'])):
    probability_value.append(b['Fitness'][i] / sum_fitness)

# Display the probability values
probability_value = [probability_value]

# Create DataFrame for dictionary b
df_b = pd.DataFrame(b)

# Create DataFrame for probability_value list
df_probability = pd.DataFrame(probability_value).transpose()
df_probability.columns = ['probability_value']

# Concatenate the two DataFrames horizontally
result_df = pd.concat([df_b, df_probability], axis=1)

expected_count = []
for i in range(len(result_df['probability_value'])):
    expected_count.append((result_df['probability_value'][i])*6)

df_expected_count = pd.DataFrame(expected_count)
df_expected_count.columns = ['expected_count']
final_roulette_wheel = pd.concat([result_df, df_expected_count], axis=1)



