import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test_enc, history_embedding):
    plt.plot(history_embedding.history['loss'], c='b', label='train')
    plt.plot(history_embedding.history['val_loss'], c='r', label='validation')
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    plt.plot(history_embedding.history['accuracy'], c='b', label='train')
    plt.plot(history_embedding.history['val_accuracy'], c='r', label='validation')
    plt.legend(loc='lower right')
    plt.xlabel('epoch')
    plt.ylabel('val accuracy')
    plt.show()

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test_enc, axis=1)
    print(classification_report(y_pred, y_true, target_names=['non_debt', 'satd'], digits=3))
