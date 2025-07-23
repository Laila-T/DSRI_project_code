import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError


def load_and_combine_features(left_file_dir, right_file_dir):

    X_lh = np.load(left_file_dir) 
    X_rh = np.load(right_file_dir)

    X_combined = np.concatenate([X_lh, X_rh], axis=2)
    return X_combined

def build_gru_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),  #shape:(timesteps, features)
        GRU(100, return_sequences=True),  
        Dropout(0.3),
        GRU(75, return_sequences = True),
        Dropout(0.3),
        GRU(50, return_sequences = True),
        Dropout(0.3),
        GRU(25, return_sequences = False), 
        Dropout(0.3),
        Dense(1) 
    ])
    
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(learning_rate=0.001),
        metrics = [tf.keras.metrics.MeanAbsoluteError()]
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

    X_train = np.concatenate([X1, X2], axis = 0)
    y_train = np.concatenate([y1, y2], axis =0)

    X_val = X3
    y_val = y3

    X_test = X4
    y_test = y4
    
    model.fit(X_train, y_train, epochs = 15, batch_size = 64, validation_data = (X_val, y_val))

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    val_predictions = model.predict(X_val)

    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test Loss (MSE): {loss}")
    print(f"Test MAE: {mae}")

    mean_train_target = np.mean(y_train)
    zero_rule_predictions = np.full_like(y_test, fill_value=mean_train_target)
    
    mae_metric = MeanAbsoluteError()
    mse_metric = MeanSquaredError()

    mae_metric.update_state(y_test, zero_rule_predictions)
    mse_metric.update_state(y_test, zero_rule_predictions)

    mae_zero_rule = mae_metric.result().numpy()
    mse_zero_rule = mse_metric.result().numpy()

    print(f"Zero-Rule Regressor MAE: {mae_zero_rule}")
    print(f"Zero-Rule Regressor MSE: {mse_zero_rule}")

    len1 = X1.shape[0]
    len2 = X2.shape[0]
    len3 = X3.shape[0]
    len4 = X4.shape[0]

    print("Predictions for 2015 Tracking Data:")
    print(train_predictions[:len1])

    print("Predictions for 2016 Session 1:")
    print(train_predictions[len1:len1+len2])

    print("Predictions for 2016 Session 2:")
    print(val_predictions[:len3])

    print("Predictions for 2016 Session 3:")
    print(test_predictions[:len4])

