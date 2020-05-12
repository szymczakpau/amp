from keras import backend


def sensitivity(y_true, y_pred):
    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + backend.epsilon())


def specificity(y_true, y_pred):
    true_negatives = backend.sum(backend.round(backend.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = backend.sum(backend.round(backend.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + backend.epsilon())


def mcc(y_true, y_pred):
    y_pred_pos = backend.round(backend.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = backend.round(backend.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = backend.sum(y_pos * y_pred_pos)
    tn = backend.sum(y_neg * y_pred_neg)

    fp = backend.sum(y_neg * y_pred_pos)
    fn = backend.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = backend.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + backend.epsilon())
