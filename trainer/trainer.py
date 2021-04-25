import argparse
from trainer import model
from trainer import dataprep

def parcero():
    parce = argparse.ArgumentParser()
    parce.add_argument('--job-dir', required=True)
    parce.add_argument('--epochs', default=10, type=int)
    parce.add_argument('--batch-size', default=32, type=int)
    return parce.parse_args()

def train_model():
    arguments = parcero()
    (x_train, y_train), (x_test, y_test) = dataprep.get_and_pad_imdb_dataset()
    imdb_word_index = dataprep.get_imdb_word_index()
    max_index_value = max(imdb_word_index.values()) # should match num_words arg in get data function

    classifier = model.get_model(vocab_size=max_index_value)
    callbacks = model.get_callbacks(arguments.job_dir)

    history = classifier.fit(x_train, y_train, epochs=arguments.epochs, batch_size=arguments.batch_size,
                            validation_data=(x_test, y_test), validation_steps=20,
                            callbacks=callbacks)


if __name__ == '__main__':
    train_model()