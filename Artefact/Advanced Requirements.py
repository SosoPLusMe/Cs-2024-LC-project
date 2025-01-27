# This program takes three values from a CSV file and compares them to predict a fourth value
# predicted_X_Value = predict_mood(A,B,C)
# print("The predicted value is", predicted_X_Value)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Training the model

# Load your dataset
data = pd.read_csv('sleepmodel.txt')

# Define your independent variables (features) and dependent variable (target)
X = data[['Times awoken','Stress Level', 'Average noise level']]
Y = data['Quality of sleep']

# Splitting the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Creating the Linear Regression model
model = LinearRegression()

# Fitting the model with the training data
model.fit(X_train, Y_train)

# Predicting mood scores for the test set
Y_pred = model.predict(X_test)


print("Model Complete!")

# Making a prediction using the model

def predict_quality(times_awoken, stress_level, AverageNL):
    df = pd.DataFrame([[times_awoken, stress_level, AverageNL]],
                      columns=['Times awoken', 'Stress Level', 'Average noise level'])
    return model.predict(df)[0]


# Let the user enter their own 3 parameters
print("")
print("USER CHOOSES 3 PARAMETERS")
timesup = int(input("Enter amount of times the user wakes up "))
slevel = float(input("Enter stress level of user 1-10 "))
avgNL = float(input("Enter average noise level"))

predicted_mood = predict_quality(timesup, slevel, avgNL)  # Example values
final_mood = round(predicted_mood,2)
print("\n The Predicted Quality of sleep for the values entered is", final_mood)


# WHAT-IF Question 1
# What if all three parameters are low?
print("-----------------------------------------------------------")
print("WHAT-IF QUESTION 1")
print("What if all three parameters are low?")

# Low values for all 3 parameters
timesup = 1
slevel = 1
avgNL = 1

sleep_low = predict_quality(timesup, slevel, avgNL)  # Example values
final_low = round(sleep_low,2)
print("\n The Quality of sleep if all three parameters are low is", final_low)



# WHAT-IF Question 2
# What if all three parameters are high?
print("-----------------------------------------------------------")
print("WHAT-IF QUESTION 2")
print("What if all three parameters are high?")

# High values for all 3 parameters
timesup = 100
slevel = 10
avgNL = 100

sleep_high = predict_quality(timesup, slevel, avgNL)  # Example values
final_high = round(sleep_high,2)
print("\n The Quality of sleep if all three parameters are high is", final_high)



# AR3 Show Results of WHAT IF on a graph for Questions 1 & 2
import matplotlib.pyplot as plt

# Data: names of the variables and their values
variable_names = ['Quality if low paremeters', 'Quality if high paremeters',]
values = [final_low, final_high]

# Creating the bar chart
plt.bar(variable_names, values)

# Adding labels and title
plt.xlabel('Parameter Type'  )
plt.ylabel('Quality of sleep')
plt.title('Bar Chart of WHAT-IF Q1, Q2 Outcomes')

# Show the plot
plt.show()

# WHAT-IF Question 3 
# What if only one parameter is high?
print("-----------------------------------------------------------")
print("WHAT-IF QUESTION 3")
print("What if only one parameter is high but the other parameters are low?")

# high value for stress level
timesup = 1
slevel = 10
avgNL = 1

stress_high = predict_quality(timesup, slevel, avgNL)
final_stress = round(stress_high,2)
print("\n The Quality of sleep if only the stress level is high", final_stress)


# high value for Times user gets up
timesup = 100
slevel = 1
avgNL = 1

ups_high = predict_quality(timesup, slevel, avgNL)
final_ups = round(ups_high,2)
print("\n The Quality of sleep if only the amount of times you get up is high is", final_ups)


# high value for average noise level
timesup = 1
slevel = 1
avgNL = 100

NL_high = predict_quality(timesup, slevel, avgNL)
final_NL = round(NL_high,2)
print("\n The Quality of sleep if only the noise level is high is", final_NL)

#Finding biggest idicator of quality of sleep (Wellbeing insight)
if  (final_stress < final_ups and final_NL):
    print("\n This shows that the stress level is the biggest indicator for finding the quality of sleep")
elif (final_ups < final_stress and final_NL):
    print("\n This shows that the amount off times you get up during your sleep is the biggest indicator for finding the quality of sleep")
elif (final_NL < final_stress and final_ups):
    print("\n This shows that the noise level is the biggest indicator for finding the quality of sleep")

# Creating the bar chart   
variable_names = ['Quality if only High Stress Level', 'Quality if only High amount of awakening','Quality if only High Average noise level']
values = [final_stress, final_ups, final_NL]

plt.bar(variable_names, values)

# Adding labels and title
plt.xlabel('Parameters'  )
plt.ylabel('Quality of sleep')
plt.title('Bar Chart of WHAT-IF Q3 Outcomes')
plt.show()
















