# # Generate Simulated Data

import numpy as np
import pandas as pd
np.random.seed(113)

timeframe = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
typeofincident = np.random.choice(["Fall", "Slips", "Equipment Malfunction"], size=len(timeframe))
severity = np.random.choice(["Low", "Medium", "High"], size=len(timeframe))
data = {
    "Date": timeframe,
    "Incident Type": typeofincident,
    "Severity": severity
}
randomdataforworkplaceinjury = pd.DataFrame(data)
randomdataforworkplaceinjury.to_csv("accident.csv", index=False)
print("data has been save successfully")


# # Exploring the generated data

readingthedata = pd.read_csv("accident.csv")
print(readingthedata)

print(readingthedata.head())
print(readingthedata.tail())
print(readingthedata.info())
print(df.describe())

print(readingthedata[['Incident Type']]. value_counts())

print(readingthedata['Severity']. value_counts())
print(readingthedata[readingthedata['Severity'] == "High"])


print(readingthedata.groupby('Incident Type').agg({'Severity': 'count'}))  # Count incidents by type


# # Data Cleaning


print(readingthedata.isnull().sum())


# No missing value. Ready for performing data analysis

# # Data Visualization


import matplotlib.pyplot as plt
import pandas as pd
readingthedata = pd.read_csv("accident.csv")


incidentcounting = readingthedata['Incident Type'].value_counts()
plt.figure(figsize=(15, 8))
plt.bar(incidentcounting.index, incidentcounting.values)
plt.xlabel('Incident Type')
plt.ylabel('Count')
plt.title('Distribution of Incident Types')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# # Identifying patterns and trends


import pandas as pd
readingthedata = pd.read_csv("accident.csv")

readingthedata.dropna(inplace=True)

readingthedata['Date'] = pd.to_datetime(readingthedata['Date'])
print(readingthedata['Date'])


# # The most common incident type

incidentcount = readingthedata.groupby('Incident Type')['Date'].count()
print(incidentcount)

import matplotlib.pyplot as plt
import numpy as np
incidentcount = readingthedata.groupby('Incident Type')['Date'].count()
colors = np.array(["red","green",'orange'])
plt.scatter(incidentcount.index, incidentcount.values, c = colors)
plt.xlabel('Incident Type')
plt.ylabel('Count')
plt.title('Incident Type vs. Count')


plt.xticks(rotation=90)
plt.show()


# ### Result: THE MOST COMMON INCIDENT TYPE
# The most common incident type is "Equipment Malfunction," followed by "Fall" and "slips"


severitymean = readingthedata.groupby('Severity')['Severity'].count()
print(severitymean)


highseverityincidents = readingthedata[readingthedata['Severity'] == 'High']
print(highseverityincidents)
highseverityincidents.to_csv("highseverity.csv", index=False)
print("Data has been saved successfully as highseverity.csv")


# ## Question: Finding out which incident type cause high severity

import pandas as pd
readingthedata = pd.read_csv("accident.csv")
severitymapping = {"Low": 1, "Medium": 2, "High": 3}
readingthedata['Severity'] = readingthedata['Severity'].map(severitymapping)
meanseverity = readingthedata['Severity'].mean()
print("Mean Severity:", meanseverity)


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
readingthedata = pd.read_csv("accident.csv")


incident = readingthedata.pivot_table(index='Incident Type', columns='Severity', values='Date', aggfunc='count')
plt.figure(figsize=(10, 6))
sns.heatmap(incident, annot=True, cmap='coolwarm')
plt.xlabel('Severity')
plt.ylabel('Incident Type')
plt.title('Incident Type vs. Severity')
plt.show()


# ### Result: Incident Type "Fall" causing most of the high severity
