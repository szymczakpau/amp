import numpy as np


def translate_generated_peptide(encoded_peptide):
    alphabet = list('ACDEFGHIKLMNPQRSTVWY')
    return ''.join([alphabet[el - 1] if el != 0 else "" for el in encoded_peptide[0].argmax(axis=1)])


def generate_unconstrained(
        n: int,
        positive: bool,
        pca: bool,
        class_selection: bool,
        decoder,
        classifier,
        pca_decomposer,
        latent_dim: int = 64,

):
    generated = []
    pos_or_neg = 1 if positive else 0
    class_min, class_max = (0.8, 1.0) if positive else (0.0, 0.2)

    while len(generated) < n:
        z = np.random.normal(size=(1, latent_dim))
        if pca:
            z = pca_decomposer.inverse_transform(z)
        z_cond = np.concatenate([z, np.array([[pos_or_neg]])], axis=1)
        decoded = decoder.predict(z_cond)
        peptide = translate_generated_peptide(decoded)
        peptide = peptide.strip("'")
        if "'" in peptide:
            continue

        class_prob = classifier.predict(np.array([decoded[0].argmax(axis=1)]))[0][0]

        if class_selection:
            if not class_min <= class_prob <= class_max:
                continue

        if (peptide, class_prob) not in generated:
            generated.append((peptide, class_prob))

    return generated
