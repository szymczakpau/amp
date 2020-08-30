from amp.models.discriminators import veltri_amp_classifier, noconv_amp_classifier
from amp.models.decoders import amp_decoder
from amp.models.encoders import amp_encoder
from amp.models.master import master


MODEL_GAREDN = {
    'VeltriAMPClassifier': veltri_amp_classifier.VeltriAMPClassifier,
    'NoConvAMPClassifier': noconv_amp_classifier.NoConvAMPClassifier,
    'AMPDecoder': amp_decoder.AMPDecoder,
    'AMPEncoder': amp_encoder.AMPEncoder,
    'MasterAMPTrainer': master.MasterAMPTrainer,
}