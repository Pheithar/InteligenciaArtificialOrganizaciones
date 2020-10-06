import numpy as np
import tensorflow as tf
import pandas as pd

csvData = np.array(pd.read_csv('../datos/time_series_covid19_confirmed_global.csv'))

real_values_spain = [748266, 769188, 778607]
real_values_brazil = [4777522, 4810935, 4847092]

model_7_days = tf.keras.models.load_model('../output/model_7_days')
model_15_days = tf.keras.models.load_model('../output/model_15_days')
model_30_days = tf.keras.models.load_model('../output/model_30_days')
model_60_days = tf.keras.models.load_model('../output/model_60_days')



spain = None
brazil = None

for row in csvData:
    if row[1] == "Spain":
        spain = row
    elif row[1] == "Brazil":
        brazil = row


spain_7_days = np.reshape(spain[-7:].astype(np.float32), (1, 7))
spain_15_days =  np.reshape(spain[-15:].astype(np.float32), (1, 15))
spain_30_days =  np.reshape(spain[-30:].astype(np.float32), (1, 30))
spain_60_days =  np.reshape(spain[-60:].astype(np.float32), (1, 60))


brazil_7_days = np.reshape(brazil[-7:].astype(np.float32), (1, 7))
brazil_15_days =  np.reshape(brazil[-15:].astype(np.float32), (1, 15))
brazil_30_days =  np.reshape(brazil[-30:].astype(np.float32), (1, 30))
brazil_60_days =  np.reshape(brazil[-60:].astype(np.float32), (1, 60))

title = "Valores reales"
title_7 = "Predicción del modelo de 7 días"
title_15 = "Predicción del modelo de 15 días"
title_30 = "Predicción del modelo de 30 días"
title_60 = "Predicción del modelo de 60 días"

def addElement(listnp, element):
    a = list(listnp[0])
    a.pop(0)
    a.append(int(element))
    return np.reshape(np.array(a), (1, 7))

def prediction(model, data):

    return model.predict(data)[0]

def predictions_3(model, data):
    pred1 = prediction(model, data)
    data = addElement(data, pred1)

    pred2 = prediction(model, data)
    data = addElement(data, pred2)

    pred3 = prediction(model, data)

    return (pred1[0], pred2[0], pred3[0])

(spain_pred1, spain_pred2, spain_pred3) = predictions_3(model_7_days, spain_7_days)

(brazil_pred1, brazil_pred2, brazil_pred3) = predictions_3(model_7_days, brazil_7_days)


print(title)

print("España día 1:", real_values_spain[0])
print("España día 2:", real_values_spain[1])
print("España día 3:", real_values_spain[2])

print("Brasil día 1:", real_values_brazil[0])
print("Brasil día 2:", real_values_brazil[1])
print("Brasil día 3:", real_values_brazil[2])

print()
print(title_7)

print("España predicción 1:", spain_pred1, "--- Error:", spain_pred1 - real_values_spain[0])
print("España predicción 2:", spain_pred2, "--- Error:", spain_pred2 - real_values_spain[1])
print("España predicción 3:", spain_pred3, "--- Error:", spain_pred3 - real_values_spain[2])

print("Brasil predicción 1:", brazil_pred1, "--- Error:", brazil_pred1 - real_values_brazil[0])
print("Brasil predicción 2:", brazil_pred2, "--- Error:", brazil_pred2 - real_values_brazil[1])
print("Brasil predicción 3:", brazil_pred3, "--- Error:", brazil_pred3 - real_values_brazil[2])
