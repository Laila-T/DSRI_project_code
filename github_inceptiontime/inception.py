# resnet model
from tensorflow import keras
import numpy as np
import time
import os  
import tensorflow as tf
from utils.utils import save_logs
from utils.utils import calculate_metrics
from utils.utils import save_test_duration

class Classifier_INCEPTION:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, batch_size=64,
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=1500):

        self.output_directory = output_directory

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs

        if build:
            self.model = self.build_model(input_shape, nb_classes)
            if verbose:
                self.model.summary()
            self.verbose = verbose

            #Use os.path.join for safe paths
            init_weights_path = os.path.join(self.output_directory, 'model_init.weights.h5') 
            self.model.save_weights(init_weights_path)

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for ks in kernel_size_s:
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=ks,
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)
        x = input_layer
        input_res = input_layer

        for d in range(self.depth):
            x = self._inception_module(x)
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = os.path.join(self.output_directory, 'best_model.keras')
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def build_feature_extractor(self, input_shape):
        input_layer = keras.layers.Input(input_shape)
        x = input_layer
        input_res = input_layer

        for d in range(self.depth):
            x = self._inception_module(x)
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        model = keras.models.Model(inputs=input_layer, outputs=x)
        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true, plot_test_acc=False):
        import tensorflow as tf
        #if len(tf.config.list_physical_devices('GPU')) == 0:
            #print('Error: No GPU available.')
            #exit()

        mini_batch_size = int(min(x_train.shape[0] / 10, 16)) if self.batch_size is None else self.batch_size

        start_time = time.time()

        hist = self.model.fit(
            x_train, y_train,
            batch_size=mini_batch_size,
            epochs=self.nb_epochs,
            verbose=self.verbose,
            validation_data=(x_val, y_val) if plot_test_acc else None,
            callbacks=self.callbacks
        )

        duration = time.time() - start_time

        #Safe model save
        last_model_path = os.path.join(self.output_directory, 'last_model.keras') #CHANGED RECENT
        self.model.save(last_model_path)

        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val, return_df_metrics=False)

        np.savetxt(os.path.join(self.output_directory, 'y_pred.txt'), y_pred)

        y_pred = np.argmax(y_pred, axis=1)
        print(hist.history.keys())

        df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration, plot_test_acc=plot_test_acc)

        keras.backend.clear_session()
        return df_metrics


    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        start_time = time.time()
        print(self.output_directory)
        model_path = os.path.join(self.output_directory, 'best_model.keras')  
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test, batch_size=self.batch_size)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(os.path.join(self.output_directory, 'test_duration.csv'), test_duration)  # CHANGED
            return y_pred
