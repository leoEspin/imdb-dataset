import argparse
import model
import dataprep

def parcero():
    parce = argparse.ArgumentParser()
    parce.add_argument('--job-dir', required=True)
    parce.add_argument('--epochs', default=10, type=int)
    parce.add_argument('--batch-size', default=32, type=int)
    parce.add_argument('--max-seq-len', default=250, type=int) # limit review size to 250 words to reduce training time
    parce.add_argument('--vocab-size', default=10000, type=int) # number of words in the model's vocabulary
    return parce.parse_args()

def train_model():
    arguments = parcero()
    (x_train, y_train), (x_test, y_test) = dataprep.get_and_pad_imdb_dataset(vocab_size=arguments.vocab_size,
                                                                            maxlen=arguments.max_seq_len)
    imdb_word_index = dataprep.get_imdb_word_index()
    max_index_value = max(imdb_word_index.values()) 

    classifier = model.get_model(vocab_size=max_index_value)
    callbacks = model.get_callbacks(arguments.job_dir)

    history = classifier.fit(x_train, y_train, epochs=arguments.epochs, batch_size=arguments.batch_size,
                            validation_data=(x_test, y_test), validation_steps=20,
                            callbacks=callbacks)


if __name__ == '__main__':
    train_model()