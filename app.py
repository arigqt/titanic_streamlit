import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import plotly.figure_factory as ff

st.title('Little exploration of the Titanic Dataset with a bit of Machine Learning')

image = Image.open('titanic.jpg')
st.image(image)

st.subheader("Let's check the data")
df = pd.read_csv('data/predictions/train_with_predictions.csv')
st.write(df)

st.subheader("Survival Pie Chart")
fig, ax = plt.subplots()
labels = ['Survived', 'Not Survived']
sizes = [len(df[df['Survived'] == 1]), len(df[df['Survived'] == 0])]
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig)

hist_values = np.histogram(df["Age"], bins=30, range=(0, 30))[0]
st.bar_chart(hist_values)


st.subheader('Distribution of Ages')
st.write('The NaN data has been filles with the mean value.')
fig, ax = plt.subplots()
ax.hist(df.Age, bins=20)
ax.set_xlabel("Ages")
ax.set_ylabel('Number of individuals')
st.pyplot(fig)


gender_survival = df.groupby(['Sex', 'Survived']).size().unstack(fill_value=0)
gender_survival['Total'] = gender_survival.sum(axis=1)
gender_survival['Survival Rate'] = gender_survival[1] / \
    gender_survival['Total']

# Create a stacked bar chart
fig, ax = plt.subplots()
gender_survival[[0, 1]].plot(kind='bar', stacked=True, ax=ax, color=[
                             'red', 'green'], alpha=0.7)
ax.set_xlabel('Sex')
ax.set_ylabel('Count')
ax.set_title('Survival Comparison by Gender')
ax.legend(['Not Survived', 'Survived'], loc='upper right')
plt.xticks(rotation=0)

# Annotate the bars with survival rate percentages
for i, rate in enumerate(gender_survival['Survival Rate']):
    plt.text(i, gender_survival['Total'][i] + 5,
             f'{rate:.2%}', ha='center', va='bottom')

# Display the chart in Streamlit
st.pyplot(fig)

# Create a Streamlit app
st.subheader("Confusion Matrix")

# Calculate confusion matrix
conf_matrix = confusion_matrix(df['Survived'], df['Predictions'])

# Create a heatmap for visualization
st.write("The raw CatBoostClassifier present correct results with a higly biaised test set (same as train)")
st.write(f"Accuracy Score: {accuracy_score(df.Survived, df.Predictions):.3f}")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False, ax=ax)
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
st.pyplot(fig)
