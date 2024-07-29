from tensorflow.keras import regularizers # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.models import Model, Sequential # type: ignore
from tensorflow.keras.layers import MaxPooling2D, Input, LSTM, Dense, Conv2D, GlobalAveragePooling2D, BatchNormalization, Dropout, Concatenate # type: ignore

def MultiModal_Model(learning_rate=0.0001, dropout_rate_cnn=0.5, dropout_rate_lstm=0.2, labels=3):
    CNN_model = Sequential()
    
    CNN_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3), padding='same'))
    CNN_model.add(MaxPooling2D(pool_size=(2, 2)))
    CNN_model.add(BatchNormalization())
    
    CNN_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    CNN_model.add(MaxPooling2D(pool_size=(2, 2)))
    CNN_model.add(BatchNormalization())
    
    CNN_model.add(GlobalAveragePooling2D())
    CNN_model.add(Dropout(dropout_rate_cnn))

    LSTM_model = Sequential()
    
    LSTM_model.add(LSTM(256, activation='relu', return_sequences=True, input_shape=(1, 178)))
    LSTM_model.add(BatchNormalization())
    LSTM_model.add(LSTM(128, activation='relu'))
    LSTM_model.add(BatchNormalization())
    LSTM_model.add(Dropout(dropout_rate_lstm))

    eeg_input = Input(shape=(1, 178))
    image_input = Input(shape=(64, 64, 3))

    cnn_output = CNN_model(image_input)
    lstm_output = LSTM_model(eeg_input)

    combined_output = Concatenate()([cnn_output, lstm_output])

    x = Dense(128, activation='relu')(combined_output)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    final_output = Dense(labels, activation='softmax')(x)

    model = Model(inputs=[eeg_input, image_input], outputs=final_output)

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def CNN_Model(learning_rate=0.0001, dropout_rate_cnn=0.5,labels=3):
    cnn_model = Sequential()
    
    cnn_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3), padding='same'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(BatchNormalization())
    
    cnn_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(BatchNormalization())
    
    cnn_model.add(GlobalAveragePooling2D())
    cnn_model.add(Dropout(dropout_rate_cnn))


    cnn_model.add(Dense(128, activation='relu'))
    cnn_model.add(Dense(64, activation='relu'))
    cnn_model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    cnn_model.add(Dense(labels, activation='softmax'))

    cnn_model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return cnn_model

def LSTM_Model(learning_rate=0.0001, dropout_rate_lstm=0.2, labels=3):
    lstm_model = Sequential()
    
    lstm_model.add(LSTM(256, activation='relu', return_sequences=True, input_shape=(1, 178)))
    lstm_model.add(BatchNormalization())
    lstm_model.add(LSTM(128, activation='relu'))
    lstm_model.add(BatchNormalization())
    lstm_model.add(Dropout(dropout_rate_lstm))

    lstm_model.add(Dense(128, activation='relu'))
    lstm_model.add(Dense(64, activation='relu'))
    lstm_model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    lstm_model.add(Dense(labels, activation='softmax'))

    lstm_model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return lstm_model