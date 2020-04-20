from amp.models.discriminators import veltri_amp_classifier
from amp.models.decoders import amp_decoder
from amp.models.encoders import amp_encoder


MODEL_GAREDN = {
    'VeltriAMPClassifier': veltri_amp_classifier.VeltriAMPClassifier,
    'AMPDecoder': amp_decoder.AMPDecoder,
    'AMPEncoder': amp_encoder.AMPEncoder,
}