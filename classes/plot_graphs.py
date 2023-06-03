import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


class GraphPlotter:
    def __init__(self, X_train, y_train, X_test, y_test, y_pred, best_model, df, save_path="log/"):
        self.X_test = X_test
        self.save_path = save_path
        self.X_train = X_train
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_pred
        self.best_model = best_model
        self.df = df

        self.plot_data_distribution()
        self.plot_confusion_matrix()

    def plot_data_distribution(self, show_plot=False):
        try:
            # plot data distribution
            plt.figure(figsize=(10, 5))
            sns.countplot(x='class', data=self.df)
            plt.title("Data Distribution")
            plt.savefig(self.save_path + "data_distribution.png")
            if show_plot:
                plt.show()
        except Exception as e:
            print(e)
            print("Error in plotting data distribution")

    def plot_confusion_matrix(self, show_plot=False):
        try:
            # plot confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(self.y_test, self.y_pred)
            plt.figure(figsize=(10, 10))
            sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
            all_sample_title = 'Accuracy Score: {:.3f}'.format(self.best_model.score(self.X_test, self.y_test))
            plt.title(all_sample_title, size=15)
            plt.savefig(self.save_path + "confusion_matrix.png")
            if show_plot:
                plt.show()
        except Exception as e:
            print(e)
            print("Error in plotting confusion matrix")

    def plot_learning_curve(self, show_plot=False):
        try:
            # plot learning curve
            from sklearn.model_selection import learning_curve
            train_sizes, train_scores, test_scores = learning_curve(self.best_model, self.X_train, self.y_train, cv=5,
                                                                    n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50),
                                                                    verbose=1)
            plt.figure(figsize=(10, 10))
            plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Training score')
            plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label='Validation score')
            plt.xlabel('Training examples', fontsize=15)
            plt.ylabel('Score', fontsize=15)
            plt.title('Learning curve', fontsize=15)
            plt.legend(loc='best')
            plt.savefig(self.save_path + "learning_curve.png")
            if show_plot:
                plt.show()
        except Exception as e:
            print(e)
            print("Learning curve not available for this model")

    # feature importance
    def plot_feature_importance(self, show_plot=False):
        try:
            importance = self.best_model.feature_importances_
            indices = np.argsort(importance)[::-1]
            plt.figure(figsize=(10, 10))
            plt.title("Feature importances")
            plt.bar(range(self.X_train.shape[1]), importance[indices], color="r", align="center")
            plt.xticks(range(self.X_train.shape[1]), indices)
            plt.xlim([-1, self.X_train.shape[1]])
            plt.savefig(self.save_path + "feature_importance.png")
            if show_plot:
                plt.show()
        except Exception as e:
            print(e)
            print("Feature importance not available for this model")

    def plot_roc_curve(self, show_plot=False):
        try:
            from sklearn.metrics import roc_curve, auc
            y_score = self.best_model.predict_proba(self.X_test)
            fpr, tpr, _ = roc_curve(self.y_test, y_score[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(10, 10))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([-0.01, 1.0])
            plt.ylim([0.0, 1.01])
            plt.xlabel('False Positive Rate', fontsize=15)
            plt.ylabel('True Positive Rate', fontsize=15)
            plt.title('Receiver operating characteristic example', fontsize=15)
            plt.legend(loc="lower right")
            plt.savefig(self.save_path + "roc_curve.png")
            if show_plot:
                plt.show()
        except Exception as e:
            print(e)
            print("ROC curve not available for this model")
