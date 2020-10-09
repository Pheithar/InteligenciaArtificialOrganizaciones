import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



BATCH_SIZE = 10
EPOCHS = 500


INPUT_SHAPE_7 = (7,)
INPUT_SHAPE_15 = (15,)
INPUT_SHAPE_30 = (30,)
INPUT_SHAPE_60 = (60,)


csvData = pd.read_csv('../datos/time_series_covid19_confirmed_global.csv')

csvData_lastDay = np.array(csvData[csvData.columns[-1]])

csvData_last7 = np.array(csvData[csvData.columns[-8:-1]])

csvData_last15 = np.array(csvData[csvData.columns[-16:-1]])

csvData_last30 = np.array(csvData[csvData.columns[-31:-1]])

csvData_last60 = np.array(csvData[csvData.columns[-61:-1]])


list_error_7_days = []

list_error_15_days = []

list_error_30_days = []

list_error_60_days = []


# Modelo 7 días

model_7_days = Sequential()
model_7_days.add(Dense(64, input_shape=INPUT_SHAPE_7, activation='relu'))
model_7_days.add(Dense(1, activation='linear'))


model_7_days.compile(loss='mean_absolute_error',
                    optimizer='adam',
                    metrics=['mean_absolute_error'])

model_7_days.fit(csvData_last7, csvData_lastDay,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  verbose=0)

model_7_days.save('../output/model_7_days')




# Modelo 15 días

model_15_days = Sequential()
model_15_days.add(Dense(64, input_shape=INPUT_SHAPE_15, activation='relu'))
model_15_days.add(Dense(1, activation='linear'))


model_15_days.compile(loss='mean_absolute_error',
                    optimizer='adam',
                    metrics=['mean_absolute_error'])

model_15_days.fit(csvData_last15, csvData_lastDay,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  verbose=0)

model_15_days.save('../output/model_15_days')



# Modelo 30 días

model_30_days = Sequential()
model_30_days.add(Dense(128, input_shape=INPUT_SHAPE_30, activation='relu'))
model_30_days.add(Dense(1, activation='linear'))


model_30_days.compile(loss='mean_absolute_error',
                    optimizer='adam',
                    metrics=['mean_absolute_error'])

model_30_days.fit(csvData_last30, csvData_lastDay,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  verbose=0)

model_30_days.save('../output/model_30_days')





# Modelo 60 días

model_60_days = Sequential()
model_60_days.add(Dense(128, input_shape=INPUT_SHAPE_60, activation='relu'))
model_60_days.add(Dense(1, activation='linear'))


model_60_days.compile(loss='mean_absolute_error',
                    optimizer='adam',
                    metrics=['mean_absolute_error'])

model_60_days.fit(csvData_last60, csvData_lastDay,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  verbose=0)

model_60_days.save('../output/model_60_days')





error_7_string = "Error total de la red de 7 días es: " + str(sum(list_error_7_days)/len(list_error_7_days))
error_15_string = "Error total de la red de 15 días es: " + str(sum(list_error_15_days)/len(list_error_15_days))
error_30_string = "Error total de la red de 30 días es: " + str(sum(list_error_30_days)/len(list_error_30_days))
error_60_string = "Error total de la red de 60 días es: " + str(sum(list_error_60_days)/len(list_error_60_days))




print(error_7_string)
print()
print(error_15_string)
print()
print(error_30_string)
print()
print(error_60_string)

with open('../output/out.txt', 'w') as file:
    file.write(error_7_string + '\n')
    file.write(error_15_string + '\n')
    file.write(error_30_string + '\n')
    file.write(error_60_string + '\n')
