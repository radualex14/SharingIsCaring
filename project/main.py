# Libraries
# Math
import pandas as pd
import numpy as np
import sys
import pprint
import operator
from itertools import islice
from collections import Counter

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot

# Loading the csv into a DataFrame
df = pd.read_csv('D:\\Python Projects\\project\\unicorn_companies.csv')

# Cleaning the data

# Renaming columns
df = pd.DataFrame(df)
df.rename(columns={'Date Joined': 'Date'}, inplace=True)
df.rename(columns={'Valuation ($B)': 'Valuation'}, inplace=True)
df.rename(columns={'Select Investors': 'Investors'}, inplace=True)
df['Investors'] = df['Investors']

# Splitting the date
date = df.Date.str.split('/', expand=True)
df['year'] = date[2]
df['month'] = date[1]
df['day'] = date[0]

df.year = pd.to_numeric(df.year)
df.month = pd.to_numeric(df.month)
df.day = pd.to_numeric(df.day)

print(df)
print()

df['Valuation'] = df['Valuation'].str.replace('$', ' ', regex=True)
df.Valuation = pd.to_numeric(df.Valuation)

# Startups by Industry
#fig = px.pie(df, names='Industry')
#fig.show()

# Year wise company joined the unicorn club

#fig2 = px.line(df, x='Company', y='year', title = 'Year the Company Joined')
#fig2.show()

#Map when company joined

for year in df.iterrows():
    ciuciu = Counter(df.year)

sorted_ciuciu = sorted(ciuciu.items(), key=operator.itemgetter(0))
print(sorted_ciuciu)
print()

labels, ys = zip(*sorted_ciuciu)
xs = np.arange(len(labels))
width = 1

fig = plt.figure()
ax = fig.gca()  #get current axes
ax.plot(xs, ys, width)

#Remove the default x-axis tick numbers and
#use tick numbers of your own choosing:
ax.set_xticks(xs)
#Replace the tick numbers with strings:
ax.set_xticklabels(labels)
#Remove the default y-axis tick numbers and
#use tick numbers of your own choosing:
ax.set_yticks([1,100,200,300,400,500,600])
plt.show()
# Top 10 Based on Valuation

company_val = df.sort_values(by='Valuation', ascending=False)
print(company_val.head(10))

top_10 = (company_val, 10)

# Countries with Unicorn Startups
#fig3 = px.pie(company_val, names='Country')
#fig3.show()


# Investors in Companies

inv_dict = {}

long_string = ""

inv_column = df.Investors
inv_column = inv_column.astype(str)
total_companies = 0
for x in range(len(inv_column)):

    long_string += inv_column[x]
    long_string += ", "
    current_companies = inv_column[x].split(", ")
    for company in current_companies:
        if company not in inv_dict:
            inv_dict[company] = total_companies
            total_companies += 1
            # print(company)
not_sorted_dict = {}
for keey in inv_dict.keys():
    invested1 = long_string.count(keey)
    #print("Number of occurrence of " + keey + ": " + str(invested1))
    not_sorted_dict[keey] = invested1

sorted_dict = sorted(not_sorted_dict.items(), key=operator.itemgetter(1), reverse = True)
#Top 10 investors
top_10 = list(sorted_dict)[:10]
print(top_10)