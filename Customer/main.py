# Libraries
#Display
from sklearn.datasets import load_iris
# Math
import pandas as pd
import numpy as np
import sys
import pprint
import operator
from itertools import islice
from collections import Counter
from datetime import date

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot

#Loading the dataframe
df = pd.read_csv('marketing_campaign.csv', sep='\t')

#Data Cleaning And Renaming

df=pd.DataFrame(df)

df.rename(columns = {'MntWines':'Wine'}, inplace = True)
df.rename(columns = {'MntFruits':'Fruit'}, inplace = True)
df.rename(columns = {'MntMeatProducts':'Meat'}, inplace = True)
df.rename(columns = {'MntFishProducts':'Fish'}, inplace = True)
df.rename(columns = {'MntSweetProducts':'Sweet'}, inplace = True)
df.rename(columns = {'MntGoldProds':'Gold'}, inplace = True)
df.rename(columns = {'NumDealsPurchases':'Deals'}, inplace = True)
df.rename(columns = {'NumWebPurchases':'Online'}, inplace = True)
df.rename(columns = {'NumCatalogPurchases':'Catalog'}, inplace = True)
df.rename(columns = {'NumStorePurchases':'Store'}, inplace = True)
df.rename(columns = {'NumWebVisitsMonth':'WebVisits'}, inplace = True)
df.rename(columns = {'AcceptedCmp3':'Campaign3'}, inplace = True)
df.rename(columns = {'AcceptedCmp4':'Campaign4'}, inplace = True)
df.rename(columns = {'AcceptedCmp5':'Campaign5'}, inplace = True)
df.rename(columns = {'AcceptedCmp1':'Campaign1'}, inplace = True)
df.rename(columns = {'AcceptedCmp2':'Campaign2'}, inplace = True)

#Clone the original dataframe
df2 = df

# Calculate Age of Customers
for i in df2['Year_Birth']:
    df2['Age'] = 2014 - df2['Year_Birth']


#Age segment

age_cond = [
    (df2['Age'] < 20),
    (df2['Age']>= 20) & (df2['Age'] <30),
    (df2['Age']>= 30) & (df2['Age'] <40),
    (df2['Age']>= 40) & (df2['Age'] <50),
    (df2['Age']>= 50) & (df2['Age'] <60),
    (df2['Age']>= 60) & (df2['Age'] <70),
    (df2['Age']>= 70) & (df2['Age'] <80),
    (df2['Age']>= 80) & (df2['Age'] <90),
    (df2['Age']>= 90) & (df2['Age'] <100),
    (df2['Age'] > 100)
]

age_values = ['Under_20', '20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s', 'Over 100s']

df2['Age_Group'] = np.select(age_cond, age_values)

#Plot Age Group
sns.set_theme(style="darkgrid")
ax = sns.countplot(x="Age_Group", data=df2)
ax.bar_label(ax.containers[0])
ax.set_title('Age Group of the Customers')
plt.show()

# PLot Age group and Type of Food

df_melted = pd.melt(df, id_vars=['Age_Group'], value_vars=['Meat', 'Fruit', 'Fish', 'Sweet', 'Gold'])

sns.set_theme(style = 'whitegrid', palette = 'pastel')
ax1 = sns.barplot(data = df_melted, x='Age_Group', y='value', hue='variable', ci=None)
ax1.set_title('Product Type by Age Group')
plt.show()

#Vegetarians Only

cond_veg = [(df2['Fruit'] != 0) & (df2['Meat'] == 0) & (df2['Fish'] == 0)]
print(np.where(cond_veg))
    #No vegetarian costomers

#Mean of shopping and Age

df_melted_shopping = pd.melt(df2, id_vars=['Age_Group'], value_vars=['Online', 'Catalog', 'Store'])
print()
print(df_melted_shopping)
sns.set_theme(style = 'ticks', palette = 'pastel')
ax2 = sns.barplot(data = df_melted_shopping, x='Age_Group', y='value', hue='variable', ci=None)
ax2.set_title('Mean of Shopping by Age Group')
plt.show()

#Sweets Consumption and Dependables

dependables = [
    (df2['Kidhome'] != 0) | (df2['Teenhome'] !=0),
    (df2['Kidhome'] == 0) & (df2['Teenhome'] == 0)
]

dependable_values = ['Dependables', 'No Dependables']
df2['Dependables'] = np.select(dependables, dependable_values)

sns.set_theme(style = 'darkgrid', palette = 'husl')
ax3 = sns.barplot(data = df2, x='Dependables', y='Sweet')
ax3.set_title('Sweet Consumption in Customers with or without Dependables')
plt.show()




