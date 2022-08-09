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

print(df)

