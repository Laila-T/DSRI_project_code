import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np

features_df = pd.read_csv('C:\\Users\\Laila\\InceptionTime\\Results_Laila\\results\\inception\\TSCMotionSenseHAR\\extracted_features.csv') #Will change to actual path for DSRI features later
num_samples = features_df.shape[0]
feature_dims = 12 #dimensions
total_features = features_df.shape[1]
assert total_features % feature_dims == 0, "Features not divisible by feature_dims"

n_timesteps = total_features // feature_dims
print("n_timesteps:", n_timesteps)

X = features_df.values.reshape(num_samples, n_timesteps, feature_dims)
print("Reshaped feature shape:", X.shape)

y = np.random.rand(num_samples) ##PLACEHOLDER ~ Will replace with actual skill targets generated

def build_gru_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),  #shape:(timesteps, features)
        GRU(32, return_sequences=False),  
        Dropout(0.3),
        Dense(1) 
    ])
    
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(learning_rate=0.001),
    )



    
    return model


model = build_gru_model(input_shape=(n_timesteps, 12))
model.summary()

model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2) #Can change specifics later

predictions = model.predict(X)
print(predictions[:5])  #Print first 5 predictions ~ I can change this later.

