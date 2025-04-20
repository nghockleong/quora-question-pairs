import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
'''
Splits q1, q2 and label into 3 separate lists
'''
def load_data(X, y):
    return X['question1'].tolist(), X['question2'].tolist(), y.tolist()

'''
Given a list of questions,
encodes each question with a position vector
which will be fed into LSTM
'''
def preprocess_text(texts, pos_vec_len, max_int_idx):
    # Simple tokenization and encoding
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = max_int_idx)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen = pos_vec_len)
    return padded_sequences, tokenizer

'''
Generates train, validation and test data to be fed into Siamese NN.
'''
def generate_train_val_test_siamese(X, y, pos_vec_len, max_int_idx):
    # Load and preprocess data
    questions1, questions2, labels = load_data(X, y)
    q1_seq, _ = preprocess_text(questions1, pos_vec_len, max_int_idx)
    q2_seq, _ = preprocess_text(questions2, pos_vec_len, max_int_idx)

    # Split data into training and testing sets
    X_train_q1, X_temp_q1, X_train_q2, X_temp_q2, y_train, y_temp = train_test_split(
        q1_seq, q2_seq, labels, test_size=0.2, random_state=42, stratify = labels
    )

    X_val_q1, X_test_q1, X_val_q2, X_test_q2, y_val, y_test = train_test_split(
        X_temp_q1, X_temp_q2, y_temp, test_size = 0.5, random_state = 42, stratify = y_temp
    )

    # Convert to TensorFlow tensors
    X_train_q1 = tf.convert_to_tensor(X_train_q1, dtype=tf.int32)
    X_test_q1 = tf.convert_to_tensor(X_test_q1, dtype=tf.int32)
    X_train_q2 = tf.convert_to_tensor(X_train_q2, dtype=tf.int32)
    X_test_q2 = tf.convert_to_tensor(X_test_q2, dtype=tf.int32)
    X_val_q1 = tf.convert_to_tensor(X_val_q1, dtype=tf.int32)
    X_val_q2 = tf.convert_to_tensor(X_val_q2, dtype=tf.int32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
    return X_train_q1, X_val_q1, X_test_q1, X_train_q2, X_val_q2, X_test_q2, y_train, y_val, y_test

'''
Given a position vector of length <input_shape>,
returns a model where the output contains a semantically meaningful layer
of neurons to represent the question (taking into account of positioning of words)
'''
def create_base_model(input_shape, max_int_idx):
    input_layer = Input(shape = input_shape)
    embedding_layer = Embedding(input_dim = max_int_idx, output_dim = 64)(input_layer)
    lstm_layer = LSTM(64, dropout = 0.2)(embedding_layer)
    dense_layer = Dense(32, activation = 'relu')(lstm_layer)
    dropout_layer = Dropout(0.2)(dense_layer)
    return Model(inputs = input_layer, outputs = dropout_layer)

'''
Siamese model allows for pairwise training
where the loss function accounts for the difference in output from
2 separate LSTMs
'''
def create_siamese_model(base_model, input_shape):
    input_left = Input(shape=input_shape)
    input_right = Input(shape=input_shape)
    
    encoded_left = base_model(input_left)
    encoded_right = base_model(input_right)
    
    distance_layer = Lambda(
        lambda tensors: tf.abs(tensors[0] - tensors[1]),
        output_shape=lambda shapes: shapes[0]
    )([encoded_left, encoded_right])
    prediction_layer = Dense(1, activation='sigmoid')(distance_layer)
    
    siamese_model = Model(inputs=[input_left, input_right], outputs=prediction_layer)
    return siamese_model