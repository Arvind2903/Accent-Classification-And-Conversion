#all_models.py

from keras.layers import GRU, Dense, LSTM, Dropout, Input, Conv2D, MaxPooling2D, BatchNormalization, Reshape, UpSampling2D, GlobalAveragePooling2D
from keras.layers import Flatten, Dropout, Activation, Add, SimpleRNN
from keras.models import Sequential, Model
from keras.initializers import glorot_uniform, LecunNormal
from keras.regularizers import l2
import tensorflow.keras.backend as K

def make_skip_connection(input, output, activation, intializer):
    x = Conv2D(128, (5, 5), activation=activation, padding='same', kernel_initializer=intializer)(input)
    print("Skip First Conv2D:", x.shape)
    x = MaxPooling2D((2, 2), padding='same')(x)
    skip = Conv2D(1, (3, 3), activation=activation, padding='same', kernel_initializer=intializer,  kernel_regularizer=l2(0.01))(input)
    print("Conv 1D after skip:", skip.shape)
    skip = Flatten()(skip)
    if len(output.shape) == 3:
        skip_dense_units = output.shape[1] * output.shape[2]
    elif len(output.shape) == 4:
        skip_dense_units = output.shape[1] * output.shape[2] * output.shape[3]
    skip = Dense(skip_dense_units, activation=activation, kernel_regularizer=l2(0.01), kernel_initializer=intializer)(skip)
    skip = Activation('tanh')(skip)
    if len(output.shape) == 3:
        skip = Reshape((output.shape[1], output.shape[2]))(skip)
    elif len(output.shape) == 4:
        skip = Reshape((output.shape[1], output.shape[2], output.shape[3]))(skip)
    print("After making skip match x's size:", skip.shape)
    
    # Skip connection from encoder
    return Add()([output, skip])  

def build_cnn(input_shape):
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # Convolutional layers
    x = Conv2D(128, (2, 2))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(256, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Flatten layer
    x = Flatten()(x)
    
    # Fully connected layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    x = Dense(input_shape[0] * input_shape[1], activation='linear')(x)
    outputs = Reshape(input_shape)(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def build_cnn_gru_dense(input_shape):
    initializer = glorot_uniform(seed=42)
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(128, (2, 2), kernel_initializer=initializer),
        BatchNormalization(),
        initializer('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(256, (3, 3), kernel_initializer=initializer),
        BatchNormalization(),
        initializer('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(256, (3, 3), padding='same', kernel_initializer=initializer),
        BatchNormalization(),
        initializer('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        # Reshape to add the timesteps dimension
        Reshape((-1, 256)),
        # Adding first GRU layer with LeCun normal initialization
        GRU(128, return_sequences=True, kernel_initializer=initializer),
        # Adding second GRU layer with LeCun normal initialization
        GRU(128, return_sequences=False, kernel_initializer=initializer),
        # Dropout before the final Dense layer
        Dropout(0.25),
        # Adding Dense layer at the end
        Conv2D(128, (2, 2), kernel_initializer=initializer),
        BatchNormalization(),
        initializer('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(256, (3, 3), kernel_initializer=initializer),
        BatchNormalization(),
        initializer('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(256, (3, 3), padding='same', kernel_initializer=initializer),
        BatchNormalization(),
        initializer('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Dense(input_shape[0] * input_shape[1], initializer='linear', kernel_initializer=initializer),  
        Reshape(input_shape)
    ])
    return model

def build_lstm(input_shape):
    inputs = Input(shape=input_shape)
    
    # Reshape input to match LSTM input format
    reshaped_inputs = Reshape((input_shape[0], input_shape[1]))(inputs)
    
    # Define LSTM layers
    lstm1 = LSTM(1024, return_sequences=True)(reshaped_inputs)
    dropout1 = Dropout(0.2)(lstm1)
    lstm2 = LSTM(1024, return_sequences=True)(dropout1)
    dropout2 = Dropout(0.2)(lstm2)
    lstm3 = LSTM(1024)(dropout2)
    dropout3 = Dropout(0.2)(lstm3)
    
    # Fully connected layers
    dense1 = Dense(2048, initializer='relu')(dropout3)
    dropout4 = Dropout(0.2)(dense1)
    dense2 = Dense(512, initializer='relu')(dropout4)
    dropout5 = Dropout(0.2)(dense2)
    
    # Output layer
    output = Dense(input_shape[0] * input_shape[1], initializer='linear')(dropout5)
    output_reshaped = Reshape(input_shape)(output)
    
    # Define the model
    model = Model(inputs=inputs, outputs=output_reshaped)
    
    return model

def build_autoencoder_gru(input_shape):
    initializer = glorot_uniform(seed=42)
    input_img = Input(shape=input_shape)
    # Encoder
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform)(input_img)
    print("Encoder Conv2D 1:", x.shape)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    print("Encoder MaxPooling2D 1:", x.shape)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform)(x)
    print("Encoder Conv2D 2:", x.shape)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    print("Encoder MaxPooling2D 2:", x.shape)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform)(x)
    print("Encoder Conv2D 3:", x.shape)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    print(x.shape)
    x = GRU(16, return_sequences=True, kernel_initializer=initializer)(Reshape((-1, 256))(x))
        # Adding second GRU layer with LeCun normal initialization
    encoded = GRU(16, return_sequences=False, kernel_initializer=initializer)(x)
    print("Encoder MaxPooling2D 3:", encoded.shape)
    
    encoded = Reshape((1,1,16))(encoded)
    # Decoder
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform)(encoded)
    print("Decoder Conv2D 1:", x.shape)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    print("Decoder UpSampling2D 1:", x.shape)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform)(x)
    print("Decoder Conv2D 2:", x.shape)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    print("Decoder UpSampling2D 2:", x.shape)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform)(x)
    print("Decoder Conv2D 3:", x.shape)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    print("Decoder UpSampling2D 3:", x.shape)
    decoded = Conv2D(1, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform)(x)
    print("Decoder Conv2D 4:", decoded.shape)
    
    # Flatten before dense layer
    x = Flatten()(decoded)
    print("Flatten:", x.shape)
    
    # Dense layer to match desired output size
    dense_units = input_shape[0] * input_shape[1] * input_shape[2]
    dense_layer = Dense(dense_units, activation='linear', kernel_initializer=glorot_uniform)(x)
    print("Dense layer:", dense_layer.shape)
    
    # Reshape output
    output_reshaped = Reshape(input_shape)(dense_layer)
    print("Output after reshaping: ", output_reshaped.shape)
    
    autoencoder = Model(input_img, output_reshaped)
    
    return autoencoder

def build_autoencoder_gru(input_shape):
    initializer = glorot_uniform(seed=42)
    input_img = Input(shape=input_shape)
    # Encoder
    x = Conv2D(1024, (3, 3), activation='gelu', padding='same', kernel_initializer=initializer)(input_img)
    print("Encoder Conv2D 1:", x.shape)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    print("Encoder MaxPooling2D 1:", x.shape)
    x = Conv2D(512, (3, 3), activation='gelu', padding='same', kernel_initializer=initializer)(x)
    print("Encoder Conv2D 2:", x.shape)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    print("Encoder MaxPooling2D 2:", x.shape)
    x = Conv2D(256, (3, 3), activation='gelu', padding='same', kernel_initializer=initializer)(x)
    print("Encoder Conv2D 3:", x.shape)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    print(x.shape)
    x = GRU(16, return_sequences=True, kernel_initializer=initializer)(Reshape((-1, 256))(x))
        # Adding second GRU layer with LeCun normal initialization
    encoded = GRU(16, return_sequences=False, kernel_initializer=initializer)(x)
    print("Encoder MaxPooling2D 3:", encoded.shape)
    
    encoded = Reshape((1,1,16))(encoded)
    
    encoded = make_skip_connection(input_img, encoded, 'gelu', initializer)
    # Decoder
    x = Conv2D(256, (3, 3), activation='gelu', padding='same', kernel_initializer=initializer)(encoded)
    print("Decoder Conv2D 1:", x.shape)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    print("Decoder UpSampling2D 1:", x.shape)
    x = Conv2D(512, (3, 3), activation='gelu', padding='same', kernel_initializer=initializer)(x)
    print("Decoder Conv2D 2:", x.shape)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    print("Decoder UpSampling2D 2:", x.shape)
    x = Conv2D(1024, (3, 3), activation='gelu', padding='same', kernel_initializer=initializer)(x)
    print("Decoder Conv2D 3:", x.shape)
    x = BatchNormalization()(x)
    decoded = UpSampling2D((2, 2))(x)
    print("Decoder UpSampling2D 3:", x.shape)
    
    # GAP for global context
    x = GlobalAveragePooling2D()(decoded)
    
    # Dense layer to match desired output size
    dense_units = input_shape[0] * input_shape[1] * input_shape[2]
    x = Dense(1024,  activation='relu', kernel_initializer=initializer)(x)
    dense_layer = Dense(dense_units, activation='relu', kernel_initializer=initializer)(x)
    print("Dense layer:", dense_layer.shape)
    
    # Reshape output
    output_reshaped = Reshape(input_shape)(dense_layer)
    print("Output after reshaping: ", output_reshaped.shape)
    
    autoencoder = Model(input_img, output_reshaped)
    
    return autoencoder


def build_autoencoder_lecunn_dropout_skip_l2(input_shape):
    input_img = Input(shape=input_shape)
    initializer = LecunNormal(seed=42)
    
    # Encoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(input_img)
    print("Encoder Conv2D 1:", x.shape)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    print("Encoder MaxPooling2D 1:", x.shape)
    x = Dropout(0.25)(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(x)
    print("Encoder Conv2D 2:", x.shape)
    x = BatchNormalization()(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    print("Encoder MaxPooling2D 2:", encoded.shape)

    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(encoded)
    print("Decoder Conv2D 1:", x.shape)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = UpSampling2D((2, 2))(x)
    print("Decoder UpSampling2D 1:", x.shape)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(x)
    print("Decoder Conv2D 2:", x.shape)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = UpSampling2D((2, 2))(x)
    print("Decoder UpSampling2D 2:", x.shape)
    x = Conv2D(1, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(x)
    
    x = make_skip_connection(input_img, x, 'relu', initializer)
    
    
    # Flatten before dense layer
    decoded = Flatten()(x)
    
    # Dense layer to match desired output size with L2 regularization
    dense_units = input_shape[0] * input_shape[1] * input_shape[2]
    dense_layer = Dense(dense_units, activation='relu', kernel_regularizer=l2(0.01), kernel_initializer=initializer)(decoded)
    
    # Reshape output
    output_reshaped = Reshape(input_shape)(dense_layer)
    print("Final Output Size:", output_reshaped.shape)
    
    autoencoder = Model(input_img, output_reshaped)
    
    return autoencoder

def build_cnn_lstm_sequence_l2(input_shape):
    input_img = Input(input_shape)
    initializer = glorot_uniform(seed=42)
    
    x = Conv2D(128, (2, 2), initializer='relu', padding='same', kernel_initializer=initializer, kernel_regularizer=l2(0.01))(input_img)
    print("Encoder Conv2D 1:", x.shape)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    print("Encoder MaxPooling2D 1:", x.shape)
    x = Dropout(x)
    activated_cnn1 = initializer('tanh')(x)
    
    x = Conv2D(128, (2, 2), initializer='relu', padding='same', kernel_initializer=initializer, kernel_regularizer=l2(0.01))(x)
    print("Encoder Conv2D 2:", x.shape)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    print("Encoder MaxPooling2D 2:", x.shape)
    x = Dropout(x)
    activated_cnn2 = initializer('tanh')
    
    lstm_input_1 = Reshape(activated_cnn1.shape[1],activated_cnn1.shape[2])(activated_cnn1)
    lstm1 = LSTM(512, kernel_initializer=initializer, kernel_regularizer=l2(0.01), return_sequences=True)(lstm_input_1)
    lstm1 = Dropout(lstm1)
    print('Output of first LSTM:', lstm1.shape)
    
    lstm_input_2 = Reshape(activated_cnn1.shape[1],activated_cnn1.shape[2])(activated_cnn2)
    lstm2 = LSTM(512, kernel_initializer=initializer, kernel_regularizer=l2(0.01), return_sequences=True)(lstm_input_2)
    lstm2 = Dropout(lstm2)
    print('Output of first LSTM:', lstm1.shape)
    
    model = Model(input=input_img, output=lstm2)
    
    return model

def build_cnn_lstm_encoder_decoder_skip_l2(input_shape):
    input_img = Input(input_shape)
    initializer = glorot_uniform(seed=42)
    
    # Encoder
    
    # First convolutional block
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=initializer, kernel_regularizer=l2(0.1))(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)
    activated_cnn1 = Conv2D(1, (3, 3), activation='relu', padding='same', kernel_initializer=initializer, kernel_regularizer=l2(0.1))(x)
    activated_cnn1 = Activation('tanh')(activated_cnn1)
    print("Output after 1st Conv layer:", activated_cnn1.shape)
    
    # Second convolutional block
    x = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer=initializer, kernel_regularizer=l2(0.1))(activated_cnn1)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)
    activated_cnn2 = Conv2D(1, (3, 3), activation='relu', padding='same', kernel_initializer=initializer, kernel_regularizer=l2(0.01))(x)
    activated_cnn2 = Activation('tanh')(activated_cnn2)
    print("Output after 2nd Conv layer:", activated_cnn2.shape)
    
    print("Shape of CNN1 output:", activated_cnn1.shape)
    print("Shape of CNN2 output:", activated_cnn2.shape)
    
    # First LSTM layer
    lstm_input_1 = Reshape((activated_cnn1.shape[1], activated_cnn1.shape[2]))(activated_cnn1)
    lstm1 = LSTM(64, kernel_initializer=initializer, kernel_regularizer=l2(0.1), return_sequences=True)(lstm_input_1)
    lstm1 = Dropout(0.4)(lstm1)
    print("Output of first encoder LSTM:", lstm1.shape)
    
    # Second LSTM layer
    lstm_input_2 = make_skip_connection(activated_cnn2, lstm1, activation='relu', intializer=initializer)
    lstm2 = LSTM(128, kernel_initializer=initializer, kernel_regularizer=l2(0.1), return_sequences=True)(lstm_input_2)
    lstm2 = Dropout(0.4)(lstm2)
    print("Output of second encoder LSTM:", lstm2.shape)
    
    encoded = GRU(8, activation='tanh', kernel_initializer=initializer, kernel_regularizer=l2(0.1), return_sequences=True)(lstm2)

    # Decoder
    
    # Third LSTM layer
    lstm3 = LSTM(128, kernel_initializer=initializer, kernel_regularizer=l2(0.1), return_sequences=True)(encoded)
    lstm3_output = Dropout(0.4)(lstm3)
    print("Output of first decoder LSTM:", lstm3.shape)
    
    # Fourth LSTM layer
    lstm4 = LSTM(64, kernel_initializer=initializer, kernel_regularizer=l2(0.1), return_sequences=True)(lstm3_output)
    lstm4_output = Dropout(0.4)(lstm4)
    lstm4_output = Reshape((lstm4_output.shape[1], lstm4_output.shape[2],1))(lstm4_output)
    print("Output of second decoder LSTM:", lstm4.shape)
    
    # Upsample the output of lstm4 before passing it to the first CNN block
    upsampled_lstm3_output = UpSampling2D((2, 2))(Reshape((lstm3_output.shape[1], lstm3_output.shape[2],1))(lstm3_output))

    # CNN layer after the fourth LSTM
    cnn_after_lstm3 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer=initializer, kernel_regularizer=l2(0.1))(upsampled_lstm3_output)
    cnn_after_lstm3 = BatchNormalization()(cnn_after_lstm3)
    cnn_after_lstm3 = Dropout(0.4)(cnn_after_lstm3)
    cnn_after_lstm3 = Conv2D(1, (2, 2), activation='relu', padding='same', kernel_initializer=initializer, kernel_regularizer=l2(0.1))(cnn_after_lstm3)
    cnn_after_lstm3 = Activation('tanh')(cnn_after_lstm3)
    
    # Upsample the output of cnn_after_lstm4 before passing it to the second CNN block
    upsampled_cnn_after_lstm4 = UpSampling2D((2, 2))(cnn_after_lstm3)

    input_to_cnn4 = make_skip_connection(upsampled_cnn_after_lstm4, lstm4_output, activation='relu', intializer=initializer)
    
    # CNN layer after the fourth LSTM
    cnn_after_lstm4_second = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=initializer, kernel_regularizer=l2(0.01))(input_to_cnn4)
    cnn_after_lstm4_second = BatchNormalization()(cnn_after_lstm4_second)
    cnn_after_lstm4_second = Dropout(0.4)(cnn_after_lstm4_second)
    cnn_after_lstm4_second = Conv2D(1, (2, 2), activation='relu', padding='same', kernel_initializer=initializer, kernel_regularizer=l2(0.01))(cnn_after_lstm4_second)
    cnn_after_lstm4_second = Activation('tanh')(cnn_after_lstm4_second)
    
    # Flatten before dense layer
    x = Flatten()(cnn_after_lstm4_second)
    print("Flatten:", x.shape)
    
    # Dense layer to match desired output size
    dense_units = input_shape[0] * input_shape[1] * input_shape[2]
    dense_layer = Dense(dense_units, activation='relu', kernel_initializer=initializer)(x)
    print("Dense layer:", dense_layer.shape)
    
    # Reshape output
    output_reshaped = Reshape(input_shape)(dense_layer)
    print("Output after reshaping: ", output_reshaped.shape)
    
    # Autoencoder model
    autoencoder = Model(input_img, output_reshaped)
    
    return autoencoder
   
def build_rnn(input_shape):
    # Define the input layer
    inputs = Input(shape=input_shape)
    initializer = glorot_uniform(seed=42)
    
    # Reshape the input for SimpleRNN
    x = Reshape((input_shape[0], input_shape[1]))(inputs)
    
    # SimpleRNN layers
    x = SimpleRNN(128, return_sequences=True, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    print(x.shape)
    
    x = SimpleRNN(256, return_sequences=True, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    print(x.shape)
    
    x = Reshape((x.shape[1], x.shape[2], 1))(x)
    
    # Flatten layer
    x = Conv2D(64, (3,3), activation='relu', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(x)
    print(x.shape)

    x = Reshape((x.shape[1], x.shape[2]))(x)
    
    # SimpleRNN layers
    x = SimpleRNN(128, return_sequences=True, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = SimpleRNN(256, return_sequences=True, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Flatten()(x)
    # Dense layer to match desired output size
    dense_units = input_shape[0] * input_shape[1] * input_shape[2]
    dense_layer = Dense(dense_units, activation='relu', kernel_initializer=initializer)(x)
    print("Dense layer:", dense_layer.shape)
    
    # Reshape output
    output_reshaped = Reshape(input_shape)(dense_layer)
    print("Output after reshaping: ", output_reshaped.shape)
    
    model = Model(inputs, output_reshaped)
    
    return model

if __name__=='__main__':
    input_shape = (20, 229, 1)
    build_rnn(input_shape)

