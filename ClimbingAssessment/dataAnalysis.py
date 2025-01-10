# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:33:59 2024

@author: nicho
"""

from plot_custom import plot_custom, makeTopLeftSpinesInvisible
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
from scipy.interpolate import interp1d
from scipy.stats import percentileofscore

lw = 2
colorNick = 'green'
colorOwen = 'blue'
colorKeith = 'red'

dataDir =       'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/sportPerformance/ClimbingAssessment/'
dataFile = 'AssessmentDataSept2022.csv'
dataFilePersonal = 'AssessmentDataPersonal.csv'

dataPersonal = pd.read_csv(dataDir + dataFilePersonal)
data = pd.read_csv(dataDir + dataFile)
data;
data.fillna("N/A", inplace=True)


dataPersonal['weight'] = dataPersonal['weight'].astype(float)/2.2
dataPersonal['height'] = dataPersonal['height'].astype(float)*2.54
dataPersonal['bmi'] = dataPersonal['weight']/dataPersonal['height']**2*100**2
dataPersonal['maxhang'] = dataPersonal['maxhang']/2.2
dataPersonal['maxhangnorm'] = 100 + 100*dataPersonal['maxhang'].astype(float)/(dataPersonal['weight'].astype(float))
dataPersonal['weightedpull'] = dataPersonal['weightedpull']/2.2
dataPersonal['maxpullnorm'] = 100 + 100*dataPersonal['weightedpull'].astype(float)/(dataPersonal['weight'].astype(float))
dataPersonal['powl'] = dataPersonal['powl'].astype(float)/25.4
dataPersonal['powr'] = dataPersonal['powr'].astype(float)/25.4

gender_palette = {'Male': 'blue', 'Female': 'red', 'N/A': 'gray', 'Other/Prefer to Not Answer': 'purple'}


def clean_column(value):
    # Replace any nonsensical values (e.g., empty strings, unrealistic values) with "N/A"
    if value in ["", "Unknown", "-", "None", "I have not pursued sport climbing goals outside in the past year"] or isinstance(value, (int, float)) and value < 0:
        return "N/A"
    return value

data = data.applymap(clean_column)

#%% cleanup boulder grades
data['max_boulder'] = data['max_boulder'].astype(str)  # Ensure column is string
data = data[data['max_boulder'].str.match(r'V*\d+|<V3')]

def extract_grade(grade):
    if grade == "<V3":
        return 2.5  # Assign a numerical value for "<V3" (e.g., 2.5 to keep it distinct but ordered)
    match = re.match(r"V(\d+)", grade)
    return int(match.group(1)) if match else None

data['BoulderGradeNumber'] = data['max_boulder'].apply(extract_grade)

data = data.sort_values(by='BoulderGradeNumber', ascending=True)

#%% cleanup sport grades 
data['max_sport'] = data['max_sport'].astype(str)  # Ensure column is string
data = data[data['max_sport'].str.match(r'5.*\d+')]

def extract_sport_grade(grade):
    # Match the pattern 5.*a/b or 5.*c/d
    match = re.match(r'5\.(\d+)([a-z])\/([a-z])', str(grade))
    if match:
        # If the pattern matches, return it in a standardized format, for example '5.xa/yb'
        return f"5.{match.group(1)}{match.group(2)}/{match.group(3)}"
    else:
        # If it doesn't match, return 'N/A'
        return "N/A"

data['SportGradeNumber'] = data['max_sport'].apply(extract_sport_grade)

def plotPersonalData(person = "Nick", category = 'height'):
    # Add a horizontal line at height 170 cm with a label
    if person == "Nick":
        color = colorNick
    elif person == 'Keith':
        color = colorKeith
    elif person == 'Owen':
        color = colorOwen
        
    plt.axhline(y=dataPersonal[dataPersonal["Name"]==person][category].values[0], color=color, linestyle='-', linewidth=lw, label=person)
    

filtered_data = data[data['max_sport'] == goal_grade]

# #%% plot boulder data
# data = data.sort_values(by='BoulderGradeNumber', ascending=True)

# plt.figure(figsize=(10, 6))
# sns.countplot(data=data, x='max_boulder',order=data['max_boulder'].unique(), palette="viridis")
# plt.title('Distribution of Boulder Grades')
# plt.xlabel('Boulder Grade')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


#%% plot stacked male and female boulder grades
data = data.sort_values(by='BoulderGradeNumber', ascending=True)

plt.figure(figsize=(8, 4))

sns.histplot(
    data=data,
    x='max_boulder',
    hue='sex',
    multiple='stack',
    palette=gender_palette,
    shrink=0.8
)

plt.title('Stacked Histogram of Boulder Grades by Gender')
plt.xlabel('Boulder Grade')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% plot stacked male and female sport grades
data = data.sort_values(by='SportGradeNumber', ascending=True)
plt.figure(figsize=(8, 4))

sns.histplot(
    data=data,
    x='max_sport',
    hue='sex',
    multiple='stack',
    palette=gender_palette,
    shrink=0.8
)

plt.title('Stacked Histogram of Sport Grades by Gender')
plt.xlabel('Sport Grade')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#%% sport
#%% violin plot showing height
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_sport'] != "N/A") & (data['height'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['height'] = dataTemp['height'].astype(float) * 2.54


plotPersonalData("Keith",'height')
plotPersonalData("Nick",'height')
plotPersonalData("Owen",'height')

sns.violinplot(
    data=dataTemp,
    x='max_sport',
    y='height',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    order=sorted(data['max_sport'].unique(), key=lambda x: (x != "N/A", x))
)

# Customize the plot
plt.ylim([130, 220])
plt.title('Distribution of Climber Heights by Sport Grade')
plt.xlabel('Sport Grade')
plt.ylabel('Height (cm)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% violin plot showing weight
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_sport'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['weight'] = dataTemp['weight'].astype(float)/2.2

plotPersonalData("Keith",'weight')
plotPersonalData("Nick",'weight')
plotPersonalData("Owen",'weight')

sns.violinplot(
    data=dataTemp,
    x='max_sport',
    y='weight',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    order=sorted(data['max_sport'].unique(), key=lambda x: (x != "N/A", x))
)

# Customize the plot
# plt.ylim([130, 220])
plt.title('Distribution of Climber Weights by Sport Grade')
plt.xlabel('Sport Grade')
plt.ylabel('Mass (kg)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#%% violin plot showing BMI
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_sport'] != "N/A") & (data['height'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['weight'] = dataTemp['weight'].astype(float)/2.2
dataTemp['height'] = dataTemp['height'].astype(float)*2.54
dataTemp['bmi'] = dataTemp['weight']/dataTemp['height']**2*100**2

plotPersonalData("Keith",'bmi')
plotPersonalData("Nick",'bmi')
plotPersonalData("Owen",'bmi')

sns.violinplot(
    data=dataTemp,
    x='max_sport',
    y='bmi',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    order=sorted(data['max_sport'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([10, 35])
plt.title('Distribution of Climber BMIs by Boulder Grade')
plt.xlabel('Sport Grade')
plt.ylabel(r'BMI (kg/m$^2$)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#%% violin plot showing ape index
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_sport'] != "N/A") & (data['span'] != "N/A") & (data['height'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['span'] = dataTemp['span'].astype(float)*2.54
dataTemp['height'] = dataTemp['height'].astype(float)*2.54
dataTemp['apeIndex'] = dataTemp['span']- dataTemp['height']

# plotPersonalData("Keith",'bmi')
# plotPersonalData("Nick",'bmi')
# plotPersonalData("Owen",'bmi')

sns.violinplot(
    data=dataTemp,
    x='max_sport',
    y='apeIndex',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    order=sorted(data['max_sport'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([-15, 15])
plt.title('Distribution of Climber Sport Grades by Ape Index')
plt.xlabel('Sport Grade')
plt.ylabel(r'Ape Index (cm)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#%% violin plot showing pullup
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_sport'] != "N/A") & (data['pullup'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['pullup'] = dataTemp['pullup'].astype(float)

# plotPersonalData("Keith",'pullup')
plotPersonalData("Nick",'pullup')
plotPersonalData("Owen",'pullup')

sns.violinplot(
    data=dataTemp,
    x='max_sport',
    y='pullup',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    order=sorted(data['max_sport'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([0, 50])
plt.title('Distribution of Climber Sport Grade by Pullups')
plt.xlabel('Sport Grade')
plt.ylabel(r'Pullups')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#%% violin plot showing pushups
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_sport'] != "N/A") & (data['pushup'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['pushup'] = dataTemp['pushup'].astype(float)

# plotPersonalData("Keith",'pullup')
plotPersonalData("Nick",'pushup')
plotPersonalData("Owen",'pushup')

sns.violinplot(
    data=dataTemp,
    x='max_sport',
    y='pushup',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    order=sorted(data['max_sport'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([0, 80])
plt.title('Distribution of Climber Sport Grade by Pushups')
plt.xlabel('Sport Grade')
plt.ylabel(r'Pushups')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% violin plot showing continuous hang 20mm
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_sport'] != "N/A") & (data['continuous'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['continuous'] = dataTemp['continuous'].astype(float)

plotPersonalData("Keith",'continuous')
plotPersonalData("Nick",'continuous')
plotPersonalData("Owen",'continuous')

sns.violinplot(
    data=dataTemp,
    x='max_sport',
    y='continuous',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    order=sorted(data['max_sport'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([0, 120])
plt.title('Distribution of Climber Sport Grade by Continuous Hang Time, 20 mm')
plt.xlabel('Sport Grade')
plt.ylabel(r'Continuous Hang Time (s)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% violin plot showing max hang 20mm
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_sport'] != "N/A") & (data['weight'] != "N/A") & (data['maxhang'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['maxhangnorm'] = 100 + 100*dataTemp['maxhang'].astype(float)/(dataTemp['weight'].astype(float))

plotPersonalData("Keith",'maxhangnorm')
plotPersonalData("Nick",'maxhangnorm')
plotPersonalData("Owen",'maxhangnorm')

sns.violinplot(
    data=dataTemp,
    x='max_sport',
    y='maxhangnorm',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    order=sorted(dataTemp['max_sport'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([100, 200])
plt.title('Distribution of Climber Sport Grade by Max 10 s Hang Weight normalized to Body Mass, 20 mm')
plt.xlabel('Sport Grade')
plt.ylabel(r'Norm Max Hang Weight (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% violin plot showing weighted pullup
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_sport'] != "N/A") & (data['weight'] != "N/A") & (data['weightedpull'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['maxpullnorm'] = 100 + dataTemp['weightedpull'].astype(float)/(dataTemp['weight'].astype(float))*100

plotPersonalData("Keith",'maxpullnorm')
plotPersonalData("Nick",'maxpullnorm')
plotPersonalData("Owen",'maxpullnorm')

sns.violinplot(
    data=dataTemp,
    x='max_sport',
    y='maxpullnorm',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    order=sorted(dataTemp['max_sport'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([100, 230])
plt.title('Distribution of Climber Sport Grade by Max Pullup Weight normalized to Body Mass')
plt.xlabel('Sport Grade')
plt.ylabel(r'Norm Max Pullup Weight (au)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% violin plot showing repeaters1 7:3
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_sport'] != "N/A") & (data['repeaters1'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['repeaters1'] = dataTemp['repeaters1'].astype(float)

plotPersonalData("Keith",'repeaters1')
plotPersonalData("Nick",'repeaters1')
plotPersonalData("Owen",'repeaters1')

sns.violinplot(
    data=dataTemp,
    x='max_sport',
    y='repeaters1',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    order=sorted(dataTemp['max_sport'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([0, 300])
plt.title('Distribution of Climber Sport Grade by 7 s hang : 3 s rest repeaters')
plt.xlabel('Sport Grade')
plt.ylabel(r'Total Hang Time (s)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#%% violin plot showing short
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_sport'] != "N/A") & (data['shortcamp'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['shortcamp'] = dataTemp['shortcamp'].astype(float)

plotPersonalData("Keith",'shortcamp')
plotPersonalData("Nick",'shortcamp')
plotPersonalData("Owen",'shortcamp')

sns.violinplot(
    data=dataTemp,
    x='max_sport',
    y='shortcamp',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    order=sorted(dataTemp['max_sport'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([0, 400])
plt.title('Distribution of Climber Sport Grade by Short Campus Repeaters')
plt.xlabel('Sport Grade')
plt.ylabel(r'Total Hang Time (s)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% violin plot showing long campus
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_sport'] != "N/A") & (data['longcamp'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['longcamp'] = dataTemp['longcamp'].astype(float)

plotPersonalData("Keith",'longcamp')
plotPersonalData("Nick",'longcamp')
plotPersonalData("Owen",'longcamp')

sns.violinplot(
    data=dataTemp,
    x='max_sport',
    y='longcamp',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    order=sorted(dataTemp['max_sport'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([0, 200])
plt.title('Distribution of Climber Sport Grade by Long Campus Repeaters')
plt.xlabel('Sport Grade')
plt.ylabel(r'Total Hang Time (s)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% violin plot showing campus reach left
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_sport'] != "N/A") & (data['powl'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['powl'] = dataTemp['powl'].astype(float)/25.4

plotPersonalData("Keith",'powl')
plotPersonalData("Nick",'powl')
plotPersonalData("Owen",'powl')

sns.violinplot(
    data=dataTemp,
    x='max_sport',
    y='powl',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    order=sorted(dataTemp['max_sport'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([0, 2])
plt.title('Distribution of Climber Sport Grade by Campus Reach Left')
plt.xlabel('Sport Grade')
plt.ylabel(r'Reach (m)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%violin plot showing campus reach right
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_sport'] != "N/A") & (data['powr'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['powr'] = dataTemp['powr'].astype(float)/25.4

plotPersonalData("Keith",'powr')
plotPersonalData("Nick",'powr')
plotPersonalData("Owen",'powr')

sns.violinplot(
    data=dataTemp,
    x='max_sport',
    y='powr',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    order=sorted(dataTemp['max_sport'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([0, 2])
plt.title('Distribution of Climber Sport Grade by Campus Reach Right')
plt.xlabel('Sport Grade')
plt.ylabel(r'Reach (m)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% boulder
#%% violin plot showing height
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_boulder'] != "N/A") & (data['height'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['height'] = dataTemp['height'].astype(float) * 2.54
dataTemp = dataTemp.sort_values(by='BoulderGradeNumber', ascending=True)


plotPersonalData("Keith",'height')
plotPersonalData("Nick",'height')
plotPersonalData("Owen",'height')

sns.violinplot(
    data=dataTemp,
    x='max_boulder',
    y='height',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    # order=sorted(data['max_boulder'].unique(), key=lambda x: (x != "N/A", x))
)

# Customize the plot
plt.ylim([130, 220])
plt.title('Distribution of Climber Heights by Boulder Grade')
plt.xlabel('Boulder Grade')
plt.ylabel('Height (cm)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% violin plot showing weight
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_boulder'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['weight'] = dataTemp['weight'].astype(float)/2.2
dataTemp = dataTemp.sort_values(by='BoulderGradeNumber', ascending=True)

plotPersonalData("Keith",'weight')
plotPersonalData("Nick",'weight')
plotPersonalData("Owen",'weight')

sns.violinplot(
    data=dataTemp,
    x='max_boulder',
    y='weight',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    # order=sorted(data['max_boulder'].unique(), key=lambda x: (x != "N/A", x))
)

# Customize the plot
# plt.ylim([130, 220])
plt.title('Distribution of Climber Weights by Boulder Grade')
plt.xlabel('Boulder Grade')
plt.ylabel('Mass (kg)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#%% violin plot showing BMI
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_boulder'] != "N/A") & (data['height'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['weight'] = dataTemp['weight'].astype(float)/2.2
dataTemp['height'] = dataTemp['height'].astype(float)*2.54
dataTemp['bmi'] = dataTemp['weight']/dataTemp['height']**2*100**2
dataTemp = dataTemp.sort_values(by='BoulderGradeNumber', ascending=True)

plotPersonalData("Keith",'bmi')
plotPersonalData("Nick",'bmi')
plotPersonalData("Owen",'bmi')

sns.violinplot(
    data=dataTemp,
    x='max_boulder',
    y='bmi',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    # order=sorted(data['max_boulder'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([10, 35])
plt.title('Distribution of Climber Boulder Grades by BMI')
plt.xlabel('Boulder Grade')
plt.ylabel(r'BMI (kg/m$^2$)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#%% violin plot showing ape index
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_boulder'] != "N/A") & (data['span'] != "N/A") & (data['height'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['span'] = dataTemp['span'].astype(float)*2.54
dataTemp['height'] = dataTemp['height'].astype(float)*2.54
dataTemp['apeIndex'] = dataTemp['span']- dataTemp['height']
dataTemp = dataTemp.sort_values(by='BoulderGradeNumber', ascending=True)

# plotPersonalData("Keith",'bmi')
# plotPersonalData("Nick",'bmi')
# plotPersonalData("Owen",'bmi')

sns.violinplot(
    data=dataTemp,
    x='max_boulder',
    y='apeIndex',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    # order=sorted(data['max_boulder'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([-15, 15])
plt.title('Distribution of Climber Boulder Grades by Ape Index')
plt.xlabel('Boulder Grade')
plt.ylabel(r'Ape Index (cm)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#%% violin plot showing pullup
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_boulder'] != "N/A") & (data['pullup'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['pullup'] = dataTemp['pullup'].astype(float)
dataTemp = dataTemp.sort_values(by='BoulderGradeNumber', ascending=True)

# plotPersonalData("Keith",'pullup')
plotPersonalData("Nick",'pullup')
plotPersonalData("Owen",'pullup')

sns.violinplot(
    data=dataTemp,
    x='max_boulder',
    y='pullup',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    # order=sorted(data['max_boulder'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([0, 50])
plt.title('Distribution of Climber Boulder Grade by Pullups')
plt.xlabel('Boulder Grade')
plt.ylabel(r'Pullups')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#%% violin plot showing pushups
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_boulder'] != "N/A") & (data['pushup'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['pushup'] = dataTemp['pushup'].astype(float)
dataTemp = dataTemp.sort_values(by='BoulderGradeNumber', ascending=True)

# plotPersonalData("Keith",'pullup')
plotPersonalData("Nick",'pushup')
plotPersonalData("Owen",'pushup')

sns.violinplot(
    data=dataTemp,
    x='max_boulder',
    y='pushup',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    # order=sorted(data['max_boulder'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([0, 80])
plt.title('Distribution of Climber Boulder Grade by Pushups')
plt.xlabel('Boulder Grade')
plt.ylabel(r'Pushups')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% violin plot showing continuous hang 20mm
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_boulder'] != "N/A") & (data['continuous'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['continuous'] = dataTemp['continuous'].astype(float)
dataTemp = dataTemp.sort_values(by='BoulderGradeNumber', ascending=True)

plotPersonalData("Keith",'continuous')
plotPersonalData("Nick",'continuous')
plotPersonalData("Owen",'continuous')

sns.violinplot(
    data=dataTemp,
    x='max_boulder',
    y='continuous',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    # order=sorted(data['max_boulder'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([0, 120])
plt.title('Distribution of Climber Boulder Grade by Continuous Hang Time, 20 mm')
plt.xlabel('Boulder Grade')
plt.ylabel(r'Continuous Hang Time (s)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% violin plot showing max hang 20mm
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_boulder'] != "N/A") & (data['weight'] != "N/A") & (data['maxhang'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['maxhangnorm'] = 100 + 100*dataTemp['maxhang'].astype(float)/(dataTemp['weight'].astype(float))
dataTemp = dataTemp.sort_values(by='BoulderGradeNumber', ascending=True)

plotPersonalData("Keith",'maxhangnorm')
plotPersonalData("Nick",'maxhangnorm')
plotPersonalData("Owen",'maxhangnorm')

sns.violinplot(
    data=dataTemp,
    x='max_boulder',
    y='maxhangnorm',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    # order=sorted(dataTemp['max_boulder'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([100, 200])
plt.title('Distribution of Climber Boulder Grade by Max 10 s Hang Weight normalized to Body Mass, 20 mm')
plt.xlabel('Boulder Grade')
plt.ylabel(r'Norm Max Hang Weight (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% violin plot showing weighted pullup
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_boulder'] != "N/A") & (data['weight'] != "N/A") & (data['weightedpull'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['maxpullnorm'] = 100 + dataTemp['weightedpull'].astype(float)/(dataTemp['weight'].astype(float))*100
dataTemp = dataTemp.sort_values(by='BoulderGradeNumber', ascending=True)


plotPersonalData("Keith",'maxpullnorm')
plotPersonalData("Nick",'maxpullnorm')
plotPersonalData("Owen",'maxpullnorm')

sns.violinplot(
    data=dataTemp,
    x='max_boulder',
    y='maxpullnorm',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    # order=sorted(dataTemp['max_boulder'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([100, 230])
plt.title('Distribution of Climber Boulder Grade by Max Pullup Weight normalized to Body Mass')
plt.xlabel('Boulder Grade')
plt.ylabel(r'Norm Max Pullup Weight (au)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% violin plot showing repeaters1 7:3
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_boulder'] != "N/A") & (data['repeaters1'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['repeaters1'] = dataTemp['repeaters1'].astype(float)
dataTemp = dataTemp.sort_values(by='BoulderGradeNumber', ascending=True)

plotPersonalData("Keith",'repeaters1')
plotPersonalData("Nick",'repeaters1')
plotPersonalData("Owen",'repeaters1')

sns.violinplot(
    data=dataTemp,
    x='max_boulder',
    y='repeaters1',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    # order=sorted(dataTemp['max_boulder'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([0, 300])
plt.title('Distribution of Climber Boulder Grade by 7 s hang : 3 s rest repeaters')
plt.xlabel('Boulder Grade')
plt.ylabel(r'Total Hang Time (s)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#%% violin plot showing short
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_boulder'] != "N/A") & (data['shortcamp'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['shortcamp'] = dataTemp['shortcamp'].astype(float)
dataTemp = dataTemp.sort_values(by='BoulderGradeNumber', ascending=True)

plotPersonalData("Keith",'shortcamp')
plotPersonalData("Nick",'shortcamp')
plotPersonalData("Owen",'shortcamp')

sns.violinplot(
    data=dataTemp,
    x='max_boulder',
    y='shortcamp',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    # order=sorted(dataTemp['max_boulder'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([0, 400])
plt.title('Distribution of Climber Boulder Grade by Short Campus Repeaters')
plt.xlabel('Boulder Grade')
plt.ylabel(r'Total Hang Time (s)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% violin plot showing long campus
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_boulder'] != "N/A") & (data['longcamp'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['longcamp'] = dataTemp['longcamp'].astype(float)
dataTemp = dataTemp.sort_values(by='BoulderGradeNumber', ascending=True)

plotPersonalData("Keith",'longcamp')
plotPersonalData("Nick",'longcamp')
plotPersonalData("Owen",'longcamp')

sns.violinplot(
    data=dataTemp,
    x='max_boulder',
    y='longcamp',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    # order=sorted(dataTemp['max_boulder'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([0, 200])
plt.title('Distribution of Climber Boulder Grade by Long Campus Repeaters')
plt.xlabel('Boulder Grade')
plt.ylabel(r'Total Hang Time (s)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% violin plot showing campus reach left
plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_boulder'] != "N/A") & (data['powl'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['powl'] = dataTemp['powl'].astype(float)/25.4
dataTemp = dataTemp.sort_values(by='BoulderGradeNumber', ascending=True)

plotPersonalData("Keith",'powl')
plotPersonalData("Nick",'powl')
plotPersonalData("Owen",'powl')

sns.violinplot(
    data=dataTemp,
    x='max_boulder',
    y='powl',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    # order=sorted(dataTemp['max_boulder'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([0, 2])
plt.title('Distribution of Climber Boulder Grade by Campus Reach Left')
plt.xlabel('Boulder Grade')
plt.ylabel(r'Reach (m)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%violin plot showing campus reach right

plt.figure(figsize=(8, 4))

# Subset and prepare data
dataTemp = data[(data['max_boulder'] != "N/A") & (data['powr'] != "N/A") & (data['weight'] != "N/A") & ((data['sex'] == "Male") | (data['sex'] == "Female"))]
dataTemp['powr'] = dataTemp['powr'].astype(float)/25.4
dataTemp = dataTemp.sort_values(by='BoulderGradeNumber', ascending=True)



plotPersonalData("Keith",'powr')
plotPersonalData("Nick",'powr')
plotPersonalData("Owen",'powr')

sns.violinplot(
    data=dataTemp,
    x='max_boulder',
    y='powr',
    hue='sex',  # Add hue for separating by gender
    palette={'Male': 'blue', 'Female': 'red'},  # Explicit color mapping
    split=True,  # Shows male and female distributions side by side in each grade
    # order=sorted(dataTemp['max_boulder'].unique(), key=lambda x: (x != "N/A", x))
)

plt.legend()

# Customize the plot
plt.ylim([0, 2])
plt.title('Distribution of Climber Boulder Grade by Campus Reach Right')
plt.xlabel('Boulder Grade')
plt.ylabel(r'Reach (m)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()