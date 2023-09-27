import pandas as pd
import os
  
names = ['year', 'month', 'day', 'dec_year', 'sn_value' , 
         'sn_error', 'obs_num', 'extra']
df = pd.read_csv('data/aapl.csv', index_col=False)
# Split the 'date' column into three new columns: 'day', 'month', 'year'
df['Month'] = df['Date'].str.split('/').str[0]
df['Day'] = df['Date'].str.split('/').str[1]
df['Year'] = df['Date'].str.split('/').str[2]
df = df.rename(columns={'Close/Last': 'Close'})

# format the date column
df['Date'] = pd.to_datetime(df['Date'])

# sort by date
df.sort_values(by='Date', inplace=True, ascending=True)




df = df[['Year','Month','Day','Date','Volume', 'Open', 'High', 'Low', 'Close']]
df = df.replace(r'^\$', '', regex=True)


print("Starting file:")
print(df[0:10])

print("Ending file:")
print(df[-10:])

df['Open'] = df['Open'].astype(float)
df['High'] = df['High'].astype(float)
df['Low'] = df['Low'].astype(float)
df['Close'] = df['Close'].astype(float)
# split into train and test sets
df_train = df[:int(0.8 * len(df))]
df_test = df[int(0.8 * len(df)):]

spots_train = df_train['Close'].tolist()
spots_test = df_test['Close'].tolist()

print("Training set has {} observations.".format(len(spots_train)))
print("Test set has {} observations.".format(len(spots_test)))

import matplotlib.pyplot as plt
plt.figure(figsize=(14, 5))
plt.plot(df_train['Date'],df_train['Close'], label='Train')
plt.plot(df_test['Date'],df_test['Close'], label='Test')
plt.xlabel('Date')
plt.ylabel('Close')
plt.legend(loc='best')
plt.show()

import numpy as np

def to_sequences(seq_size, obs):
    x = []
    y = []

    for i in range(len(obs)-SEQUENCE_SIZE):
        #print(i)
        window = obs[i:(i+SEQUENCE_SIZE)]
        after_window = obs[i+SEQUENCE_SIZE]
        window = [[x] for x in window]
        #print("{} - {}".format(window,after_window))
        x.append(window)
        y.append(after_window)
        
    return np.array(x),np.array(y)
    
    
SEQUENCE_SIZE = 5
x_train,y_train = to_sequences(SEQUENCE_SIZE,spots_train)
x_test,y_test = to_sequences(SEQUENCE_SIZE,spots_test)

print("Shape of training set: {}".format(x_train.shape))
print("Shape of test set: {}".format(x_test.shape))

from tensorflow import keras
from tensorflow.keras import layers

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x) # project back to shape of embedding
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)

input_shape = x_train.shape[1:]

model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss="mean_squared_error",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4)
)
#model.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=10, \
    restore_best_weights=True)]

model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=64,
    callbacks=callbacks,
)

model.evaluate(x_test, y_test, verbose=1)

from sklearn import metrics

pred = model.predict(x_test)
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print("Score (RMSE): {}".format(score))
print(pred)

plt.figure(figsize=(10,6))

# Plotting the actual future values
plt.plot(y_test, label='True Future Values', color='blue')

# Plotting the predicted future values
plt.plot(pred, label='Predicted Future Values', color='red', linestyle='dashed')

plt.title('Time Series Prediction')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()




