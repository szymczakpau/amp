from keras import backend
from keras import metrics


def kl_loss(z_mean, z_sigma):
    return - 0.5 * backend.sum(
        1 + z_sigma - backend.square(z_mean) - backend.exp(z_sigma),
        axis=-1
    )


def sparse_categorical_accuracy(y_true, y_pred):
    return backend.mean(backend.equal(y_true, backend.cast(backend.argmax(y_pred, axis=-1), backend.floatx())))


def reconstruction_loss(y_true, y_pred):
    return backend.mean(metrics.sparse_categorical_crossentropy(y_true, y_pred), axis=-1)
