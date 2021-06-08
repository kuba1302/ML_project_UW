from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


class PlotRoc:

    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 9))

        self.lw = 2
        self.ax.plot([0, 1], [0, 1], color='navy', lw=self.lw, linestyle='--')

        self.ax.set_xlim([-0.01, 1.0])
        self.ax.set_ylim([0.0, 1.01])
        self.ax.set_xlabel('False Positive Rate')
        self.ax.set_ylabel('True Positive Rate')
        self.ax.set_title(f'Receiver operating characteristic')
        self.ax.legend(loc="lower right")

    def add_model(self, true, preds, label):
        fpr, tpr, thresholds = roc_curve(true, preds)
        roc_score = roc_auc_score(true, preds)
        self.ax.plot(fpr, tpr, lw=self.lw, label=f'{label} : {roc_score}')



