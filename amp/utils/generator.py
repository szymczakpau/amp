from itertools import cycle, islice

import numpy as np

def array_generator(array, batch_size):
    c = cycle(array)
    while True:
        yield np.array(list(islice(c, batch_size)))


def concatenated_generator(
        uniprot_x,
        uniprot_y,
        amp_x,
        amp_y,
        batch_size,
):
    amp_x_gen = array_generator(amp_x, batch_size)
    amp_y_gen = array_generator(amp_y, batch_size)
    uniprot_x_gen = array_generator(uniprot_x, batch_size)
    uniprot_y_gen = array_generator(uniprot_y, batch_size)

    while True:
        batch_amp_x = next(amp_x_gen)
        batch_amp_y = next(amp_y_gen)

        batch_uniprot_x = next(uniprot_x_gen)
        batch_uniprot_y = next(uniprot_y_gen)

        result_x = np.concatenate([batch_amp_x, batch_uniprot_x])
        result_amp = np.concatenate([batch_amp_y, batch_uniprot_y])

        noise_in = np.random.normal(0, 0.1, size=(result_amp.shape[0], 64))

        yield [
                  result_x,
                  result_amp,
                  noise_in,
              ], \
              [
                  result_amp,
                  np.zeros_like(result_amp),
              ]
