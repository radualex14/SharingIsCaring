# Libraries
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
    df2['Age'] = 2022 - df2['Year_Birth']

#Age segment

for item in df2['Age']:
    df2['Age_Group'] = df2['Age'] - (df2['Age'] % 10)

#Plot Age Group
sns.set_theme(style="darkgrid")
ax = sns.countplot(x="Age_Group", data=df2)
ax.bar_label(ax.containers[0])
ax.set_title('Age Group of the Customers')
plt.show()

# PLot Age group and Type of Food

df_melted = pd.melt(df, id_vars=['Age_Group'], value_vars=['Meat', 'Fish', 'Sweet', 'Gold'])
print(df_melted)
sns.set_theme(style = 'whitegrid', palette = 'pastel')
ax1 = sns.barplot(data = df_melted, x='Age_Group', y='value', hue='variable', ci=None)
ax1.set_title('Product Type by Age Group')
plt.show()


