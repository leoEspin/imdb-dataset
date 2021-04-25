import tensorflow as tf
import os

def get_model(vocab_size, emb_dim: int = 16):
    """
    Binary classification model for the IMDB dataset of 
    movie reviews
    """
    model = tf.keras.Sequential() 
    model.add(tf.keras.layers.Embedding(input_dim=vocab_size +1, 
                                        output_dim=emb_dim,
                                        mask_zero=True))
    model.add(tf.keras.layers.LSTM(units=emb_dim))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model

def get_callbacks(job_dir: str, mmetric: str = 'val_accuracy')-> list:
    """
    Creates Keras callbacks for model training.
    Keras callbacks support writting directly to cloud storage.
    """

    callbacks = []

    logs_dir = os.path.join(job_dir, 'tensorboard')
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        logs_dir, histogram_freq=1, write_graph=False, update_freq='epoch')
    callbacks.append(tensorboard_cb)

    save_dir = os.path.join(job_dir, 'checkpoints')
    best = tf.keras.callbacks.ModelCheckpoint(save_dir, save_best_only=True,
        save_weights_only=False, monitor='val_accuracy')
    callbacks.append(best)
    
    kill = tf.keras.callbacks.EarlyStopping(monitor=mmetric, patience=5)
    callbacks.append(kill)

    return callbacks