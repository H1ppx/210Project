import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle as cPickle

# Evaluate the model
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print("Test loss:", score)

# Plot training/validation loss
plt.figure()
plt.title("Training Performance")
plt.plot(history.epoch, history.history['loss'], label='Train Loss')
plt.plot(history.epoch, history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha='right')
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

# Confusion matrix for entire test set
test_Y_hat = model.predict(X_test, batch_size=batch_size)
conf = np.zeros((len(classes), len(classes)))

for i in range(X_test.shape[0]):
    true_label = np.argmax(Y_test[i])
    pred_label = np.argmax(test_Y_hat[i])
    conf[true_label, pred_label] += 1

confnorm = conf / conf.sum(axis=1, keepdims=True)
plt.figure()
plot_confusion_matrix(confnorm, labels=classes, title="Normalized Confusion Matrix")

# Accuracy by SNR
acc = {}
test_SNRs = np.array([lbl[i][1] for i in test_idx])

for snr in snrs:
    # Filter test samples by current SNR
    idx_snr = np.where(test_SNRs == snr)[0]
    X_snr = X_test[idx_snr]
    Y_snr = Y_test[idx_snr]

    Y_hat_snr = model.predict(X_snr, batch_size=batch_size)

    conf_snr = np.zeros((len(classes), len(classes)))
    for i in range(X_snr.shape[0]):
        true = np.argmax(Y_snr[i])
        pred = np.argmax(Y_hat_snr[i])
        conf_snr[true, pred] += 1

    confnorm_snr = conf_snr / conf_snr.sum(axis=1, keepdims=True)

    plt.figure()
    plot_confusion_matrix(confnorm_snr, labels=classes, title=f"Confusion Matrix (SNR = {snr})")

    correct = np.trace(conf_snr)
    total = np.sum(conf_snr)
    acc[snr] = correct / total
    print(f"SNR {snr} - Accuracy: {acc[snr]:.4f}")

# Save accuracy results
print("SNR-wise Accuracy:", acc)
with open('results_cnn2_d0.5.dat', 'wb') as fd:
    cPickle.dump(("CNN2", 0.5, acc), fd)

# Plot accuracy vs. SNR
plt.figure()
plt.plot(snrs, [acc[snr] for snr in snrs], marker='o')
plt.xlabel("Signal-to-Noise Ratio (SNR)")
plt.ylabel("Classification Accuracy")
plt.title("CNN2 Accuracy on RadioML 2016.10a")
plt.grid(True)
plt.tight_layout()
plt.show()
