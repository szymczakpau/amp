import numpy as np
import pandas as pd
from amp.data_utils import sequence

def translate_generated_peptide(encoded_peptide):
    alphabet = list('ACDEFGHIKLMNPQRSTVWY')
    return ''.join([alphabet[el - 1] if el != 0 else "" for el in encoded_peptide[0].argmax(axis=1)])

def translate_peptide(encoded_peptide):
    alphabet = list('ACDEFGHIKLMNPQRSTVWY')
    return ''.join([alphabet[el-1] if el != 0 else "" for el in encoded_peptide])

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
    counter = 0
    while len(generated) < n:
        counter += 1
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

    print(f'Generated {counter} peptides, {n} passed')
    return generated


def improve_peptides(
        input_sequences,
        classifier_model,
        encoder_model,
        decoder_model,
):
    improved = []
    originals = []
    original_raw = input_sequences
    original_padded = sequence.pad(sequence.to_one_hot(original_raw))
    original_encoded = encoder_model.predict(original_padded)
    original_probs = classifier_model.predict(original_padded)
    for z, org_peptide, org_prob in zip(original_encoded, original_raw, original_probs):
        z_cond = np.concatenate([[z], np.array([[1]])], axis=1)
        decoded = decoder_model.predict(z_cond)
        peptide = translate_generated_peptide(decoded)
        peptide = peptide.strip("'")
        if "'" in peptide or peptide == '':
            continue
        if peptide == org_peptide:
            continue

        prob = classifier_model.predict(np.array([decoded[0].argmax(axis=1)]))[0][0]
        if 0.8 <= prob <= 1.0:
            improved.append((peptide, prob))
            originals.append((org_peptide, org_prob[0]))

    print(f'Improved {len(improved)} peptides, {len(input_sequences)-len(improved)} unimproved')

    return originals, improved


def improve_single_peptide(
        template,
        positive: bool,
        df_name: str,
        classifier_model,
        encoder_model,
        decoder_model,

):
    amp_input = 1 if positive else 0

    sequence = []
    amp_prob = []

    org_amp = classifier_model.predict(template.reshape(1, 25))
    org_z = encoder_model.predict(template.reshape(1, 25))

    class_min, class_max = (0.8, 1.0) if positive else (0.0, 0.2)

    sequence.append('ORIGINAL_PEPTIDE:' + translate_peptide(template))
    amp_prob.append(org_amp[0][0])
    for i in range(100):
        for mean in [0, 0.01, 0.1]:
            for std in [0, 0.01, 0.1]:
                z = org_z + np.random.normal(mean, std, org_z.shape[0])
                sample = np.concatenate([z, np.array([[amp_input]])], axis=1)
                candidate = decoder_model.predict(sample)
                predicted_class = classifier_model.predict(np.array([candidate[0].argmax(axis=1)]))

                if not class_min <= predicted_class[0][0] <= class_max:
                    continue
                if translate_generated_peptide(candidate) in sequence:
                    continue
                if translate_generated_peptide(candidate) == translate_peptide(template):
                    continue

                sequence.append(translate_generated_peptide(candidate))
                amp_prob.append(predicted_class[0][0])

    df = pd.DataFrame.from_dict(
        {
            'sequence': sequence,
            'amp_prob': amp_prob,
        }
    )
    return df
