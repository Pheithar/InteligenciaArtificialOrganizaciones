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

size_train = int(len(csvData_lastDay) * 0.7)
size_test = int(len(csvData_lastDay) - size_train)

csvData_training_Y = csvData_lastDay[:size_train]
csvData_test_Y = csvData_lastDay[size_train:]

print()
print()
print()

# Modelo 7 días
print("Generando modelo con 7 días...")
csvData_last7_training_X = csvData_last7[:size_train]
csvData_last7_test_X = csvData_last7[size_train:]

model_7_days = Sequential()
model_7_days.add(Dense(64, input_shape=INPUT_SHAPE_7, activation='relu'))
model_7_days.add(Dense(1, activation='linear'))


model_7_days.compile(loss='mean_absolute_error',
                    optimizer='adam',
                    metrics=['mean_absolute_error'])

model_7_days.fit(csvData_last7_training_X, csvData_training_Y,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  validation_data=(csvData_last7_test_X, csvData_test_Y),
                  verbose=0)

_, test_err_7_days = model_7_days.evaluate(csvData_last7_test_X, csvData_test_Y, verbose=0)

model_7_days.save('../output/model_7_days_split')



# Modelo 15 días
print("Generando modelo con 15 días...")
csvData_last15_training_X = csvData_last15[:size_train]
csvData_last15_test_X = csvData_last15[size_train:]

model_15_days = Sequential()
model_15_days.add(Dense(64, input_shape=INPUT_SHAPE_15, activation='relu'))
model_15_days.add(Dense(1, activation='linear'))


model_15_days.compile(loss='mean_absolute_error',
                    optimizer='adam',
                    metrics=['mean_absolute_error'])

model_15_days.fit(csvData_last15_training_X, csvData_training_Y,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  validation_data=(csvData_last15_test_X, csvData_test_Y),
                  verbose=0)

_, test_err_15_days = model_15_days.evaluate(csvData_last15_test_X, csvData_test_Y, verbose=0)

model_15_days.save('../output/model_15_days_split')



# Modelo 30 días
print("Generando modelo con 30 días...")
csvData_last30_training_X = csvData_last30[:size_train]
csvData_last30_test_X = csvData_last30[size_train:]

model_30_days = Sequential()
model_30_days.add(Dense(128, input_shape=INPUT_SHAPE_30, activation='relu'))
model_30_days.add(Dense(1, activation='linear'))


model_30_days.compile(loss='mean_absolute_error',
                    optimizer='adam',
                    metrics=['mean_absolute_error'])

model_30_days.fit(csvData_last30_training_X, csvData_training_Y,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  validation_data=(csvData_last30_test_X, csvData_test_Y),
                  verbose=0)

_, test_err_30_days = model_30_days.evaluate(csvData_last30_test_X, csvData_test_Y, verbose=0)

model_30_days.save('../output/model_30_days_split')



# Modelo 60 días
print("Generando modelo con 60 días...")
csvData_last60_training_X = csvData_last60[:size_train]
csvData_last60_test_X = csvData_last60[size_train:]

model_60_days = Sequential()
model_60_days.add(Dense(128, input_shape=INPUT_SHAPE_60, activation='relu'))
model_60_days.add(Dense(1, activation='linear'))


model_60_days.compile(loss='mean_absolute_error',
                    optimizer='adam',
                    metrics=['mean_absolute_error'])

model_60_days.fit(csvData_last60_training_X, csvData_training_Y,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  validation_data=(csvData_last60_test_X, csvData_test_Y),
                  verbose=0)

_, test_err_60_days = model_60_days.evaluate(csvData_last60_test_X, csvData_test_Y, verbose=0)

model_60_days.save('../output/model_60_days_split')





error_7_string = "Error de la red de 7 días es: " + str(test_err_7_days)
error_15_string = "Error de la red de 15 días es: " + str(test_err_15_days)
error_30_string = "Error de la red de 30 días es: " + str(test_err_30_days)
error_60_string = "Error de la red de 60 días es: " + str(test_err_60_days)




print(error_7_string)
print()
print(error_15_string)
print()
print(error_30_string)
print()
print(error_60_string)

with open('../output/out_split.txt', 'w') as file:
    file.write(error_7_string + '\n')
    file.write(error_15_string + '\n')
    file.write(error_30_string + '\n')
    file.write(error_60_string + '\n')
