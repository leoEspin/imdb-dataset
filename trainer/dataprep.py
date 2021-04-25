import tensorflow as tf
import tensorflow.keras.datasets.imdb as imdb

def get_and_pad_imdb_dataset(num_words=10000, maxlen=None, index_from=3):
    from tensorflow.keras.datasets import imdb

    # Load data with defaults https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb/load_data
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words,
                                                          skip_top=0,
                                                          maxlen=maxlen,
                                                          start_char=1,
                                                          oov_char=2,
                                                          index_from=index_from)
    # pad data with defaults https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,
                                                        maxlen=maxlen,
                                                        padding='pre',
                                                        truncating='pre',
                                                        value=0)
    
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,
                                                           maxlen=maxlen,
                                                           padding='pre',
                                                           truncating='pre',
                                                           value=0)
    return (x_train, y_train), (x_test, y_test)


def get_imdb_word_index(num_words=10000, index_from=3):
    imdb_word_index = tf.keras.datasets.imdb.get_word_index()
    imdb_word_index = {key: value + index_from for
                       key, value in imdb_word_index.items() if value <= num_words-index_from}
    return imdb_word_index