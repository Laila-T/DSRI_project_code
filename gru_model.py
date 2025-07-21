import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

def load_and_combine_features(left_file_dir, right_file_dir):
    X_lh = np.load(left_file_dir) 
    X_rh = np.load(right_file_dir)
    X_combined = np.concatenate([X_lh, X_rh], axis=2)
    return X_combined

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

def load_targets(target_file_path):
    return np.load(target_file_path)

if __name__ == '__main__':

    X1= load_and_combine_features('C:\\Users\\Laila\\InceptionTime\\cvc_extracted_features\\2015_data\\features_LH.npy', 
                                       'C:\\Users\\Laila\\InceptionTime\\cvc_extracted_features\\2015_data\\features_RH.npy')
    X2= load_and_combine_features('C:\\Users\\Laila\\InceptionTime\\cvc_extracted_features\\2016_Session1\\features_LH.npy', 
                                       'C:\\Users\\Laila\\InceptionTime\\cvc_extracted_features\\2016_Session1\\features_RH.npy')
    X3= load_and_combine_features('C:\\Users\\Laila\\InceptionTime\\cvc_extracted_features\\2016_Session2\\features_LH.npy', 
                                       'C:\\Users\\Laila\\InceptionTime\\cvc_extracted_features\\2016_Session2\\features_RH.npy')
    X4 = load_and_combine_features('C:\\Users\\Laila\\InceptionTime\\cvc_extracted_features\\2016_Session3\\features_LH.npy', 
                                       'C:\\Users\\Laila\\InceptionTime\\cvc_extracted_features\\2016_Session3\\features_RH.npy')
    
    y1 = load_targets('C:\\Users\\Laila\\InceptionTime\\targets_2015.npy')
    y2 = load_targets('C:\\Users\\Laila\\InceptionTime\\targets_2016_Session1.npy')
    y3 = load_targets('C:\\Users\\Laila\\InceptionTime\\targets_2016_Session2.npy')
    y4 = load_targets('C:\\Users\\Laila\\InceptionTime\\targets_2016_Session3.npy')

    #Concatenate all data along num_sample axis
    X_all = np.concatenate([X1, X2, X3, X4], axis=0)
    y_all = np.concatenate([y1,y2,y3,y4], axis=0)

    print(f'X_all shape: {X_all.shape}, y_all shape: {y_all.shape}')

    n_timesteps = X_all.shape[1]
    feature_dims = X_all.shape[2]

    model = build_gru_model(input_shape=(n_timesteps, feature_dims))
    model.summary()

    model.fit(X_all, y_all, epochs=10, batch_size=32, validation_split=0.2)

    predictions = model.predict(X_all)
    print('predictions shape:', predictions.shape)
    print(X1.shape[0], y1.shape[0])  #Debugging
    print(X2.shape[0], y2.shape[0])  

    len1 = X1.shape[0]
    len2 = X2.shape[0]
    len3 = X3.shape[0]
    len4 = X4.shape[0]

    print("Predictions for 2015 Tracking Data:")
    print(predictions[:len1])

    print("Predictions for 2016 Session 1:")
    print(predictions[len1:len1+len2])

    print("Predictions for 2016 Session 2:")
    print(predictions[len1+len2:len1+len2+len3])

    print("Predictions for 2016 Session 3:")
    print(predictions[len1+len2+len3:])

