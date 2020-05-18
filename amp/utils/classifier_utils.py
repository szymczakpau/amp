from keras import callbacks
from keras import backend
import sklearn.metrics as metrics
import numpy as np

""" 
From Bioninja Challange:
https://github.com/jludwiczak/GrumpyPeptides 
"""


class ClassifierLogger(callbacks.Callback):

    def __init__(
            self,
            out_path='./',
            patience=10,
            lr_patience=3,
            out_fn='',
            log_fn=''
    ):
        self.f1 = 0
        self.path = out_path
        self.fn = out_fn
        self.patience = patience
        self.lr_patience = lr_patience
        self.no_improve = 0
        self.no_improve_lr = 0

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        cv_pred = self.model.predict(self.validation_data[0], batch_size=1024)
        cv_true = self.validation_data[1]
        cv_pred_bin = np.where(cv_pred > 0.5, 1, 0)
        f1_val = metrics.f1_score(cv_true, cv_pred_bin)
        auc_val = metrics.roc_auc_score(cv_true, cv_pred)
        if self.f1 < f1_val:
            self.no_improve = 0
            self.no_improve_lr = 0
            print("Epoch %s - best F1: %s, AUC %s" % (epoch, round(f1_val, 4), round(auc_val, 4)))
            self.f1 = f1_val
            self.model.save(self.path + self.fn, overwrite=True)
        else:
            self.no_improve += 1
            self.no_improve_lr += 1
            print("Epoch %s - current F1: %s, AUC: %s" % (epoch, round(f1_val, 4), round(auc_val, 4)))
            if self.no_improve >= self.patience:
                self.model.stop_training = True
            if self.no_improve_lr >= self.lr_patience:
                lr = float(backend.get_value(self.model.optimizer.lr))
                backend.set_value(self.model.optimizer.lr, 0.75 * lr)
                print("Setting lr to {}".format(0.75 * lr))
                self.no_improve_lr = 0

        return
