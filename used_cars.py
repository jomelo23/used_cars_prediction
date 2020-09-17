import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

DATA_URL = ("with_loc.csv")

st.title('Used Vehicles Sales in Southern Ontario')
st.markdown('### This app is a Streamlit Dashboard which can be used to analyze Used Car Sales in Ontario')


data = pd.read_csv(DATA_URL)
data = data.drop('Unnamed: 0',axis=1)
data = data.drop_duplicates()

mapbox_token="your_mab_box_token"

df_map = pd.read_csv("map_cleaned.csv")
midpoint = (np.average(df_map['latitude']), np.average(df_map['longitude']))

layer = pdk.Layer(
	'HexagonLayer',
	data = df_map[['latitude','longitude']],
	get_position = ['longitude', 'latitude'],
	radius  = 1500,
	extruded = True,
	pickable = True,
	elevation_scale = 100,
	elevation_range = [0,1000])

initial_view = pdk.ViewState(
	latitude=midpoint[0],
	longitude=midpoint[1],
	zoom=8,
	pitch=30,
	bearing=5,)

st.write(pdk.Deck(
	map_style='mapbox://styles/mapbox/streets-v9',
	initial_view_state=initial_view,
	layers = [layer],
	tooltip = {
		'html':'<b>Number of Cars on Sale:<b> {elevationValue}',
		'style': {
			'color':'white'
		}
		}
	)
)

st.subheader('Price Distribution before and after removing outliers in the dataset')

Q1 = data.quantile(0.05)
Q3 = data.quantile(0.89)
IQR = Q3 - Q1
data_out = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]

fig, axes = plt.subplots(1,2)

plt.figure(figsize=(9, 12))
sns.set_style('whitegrid')
bx1 = sns.boxplot(y=data['Price'],ax=axes[0]).set(
	ylabel='($) Price')

plt.figure(figsize=(9, 12))
sns.set_style('whitegrid')
bx2 = sns.boxplot(y=data_out['Price'],ax=axes[1]).set(
	ylabel='($) Price')

st.pyplot(fig)

# sns.set(font_scale = 2)
sns.set_style('whitegrid')
plt.subplots(figsize=[20,15])
plt.title('Distribution of Car Prices by Car Make', fontsize = 36)
plt.xlabel('Price', fontsize = 36)
plt.ylabel('Make', fontsize = 36)
sns.boxplot(data=data_out, x='Price', y='Make', showfliers=False)
st.pyplot()

sns.set_style('whitegrid')
plt.subplots(figsize=[20,15])
plt.title('Distribution of Car Prices by Year', fontsize = 36)
plt.xlabel('Year', fontsize = 36)
plt.ylabel('Price', fontsize = 36)
sns.boxplot(data=data_out, x='Year', y='Price', showfliers=False)
st.pyplot()

st.subheader('Use Regression to predict the price of a Car')
make = sorted(list(data_out['Make'].unique()))
model = list(data_out['Model'].unique())
year = [x+1 for x in range(1980,2020)]
location = sorted(list(data_out['Location'].unique()))

select1 = st.sidebar.selectbox('Select the Make of the Vehicle', make)
model2 = data_out.loc[data['Make'].isin([select1])]['Model'].unique()
select2 = st.sidebar.selectbox('Select the Model of the Vehicle', model2)
select3 = st.sidebar.selectbox('Select the Year of the Vehicle', year)
select4 = st.sidebar.number_input('Choose the Maximum Mileage')

for i,v in enumerate(model):
	if (
		select2 == model[i]
		):
		st.write(data_out.loc[data_out['Model'].isin([v])][['Price','Make','Model','Year','Mileage','Location','TO_distance']]) #.sort_values(by=['injured_pedestrians'], ascending = False).dropna(how='any')[:5])

select5 = st.sidebar.selectbox('Select the Location of the Vehicle', location)
select6 = st.sidebar.number_input('Choose the Distance From Toronto')

s = {'Make':select1,'Model':select2,'Year':select3,'Mileage':select4, 'Location':select5,'TO_distance':select6,'Location_Region':select5 + ', Ontario'}
selection = pd.DataFrame(data=s, index = [0])

data_out = data_out.append(selection)
data_out = data_out.reset_index().drop('index',axis=1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df3 = pd.get_dummies(data_out)
selection = df3.iloc[[-1]]
selection.drop('Price', axis=1,inplace=True)
df3.drop(df3.tail(1).index,inplace=True)
X = df3.drop('Price',axis=1)
y = df3['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) 
lr = LinearRegression()
lr.fit(X_train,y_train)
predictions = lr.predict(X_test)

sns.set_style('whitegrid')
plt.subplots(figsize=[20,15])
plt.xlabel('Test Values', fontsize = 36)
sns.jointplot(x=y_test, y=predictions, color = '#538DAA', kind = 'reg',
              height = 10)
st.pyplot()

predict_selection = lr.predict(selection)
if predict_selection <  2000:
	st.markdown('The Market Value for this car in the Area is less than $2000')
else:
	st.markdown('The Market Value for this car in the Area is $%i' % (predict_selection))
