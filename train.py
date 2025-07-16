import numpy as np
import pickle as cPickle
from keras.models import Sequential
from keras.layers import Reshape, ZeroPadding2D, Conv2D, Dropout, Flatten, Dense, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Load the dataset
# Make sure "RML2016.10a_dict.dat" is downloaded or generated beforehand
Xd = cPickle.load(open("RML2016.10a_dict.dat", 'rb'))
snrs, mods = [sorted(set(k[i] for k in Xd)) for i in [1, 0]]

# Prepare data and labels
X = []
labels = []
for mod in mods:
    for snr in snrs:
        samples = Xd[(mod, snr)]
        X.append(samples)
        labels.extend([(mod, snr)] * samples.shape[0])
X = np.vstack(X)

# Split into train/test sets
np.random.seed(2016)
n_examples = X.shape[0]
n_train = int(n_examples * 0.5)
all_indices = np.arange(n_examples)
train_idx = np.random.choice(all_indices, size=n_train, replace=False)
test_idx = np.setdiff1d(all_indices, train_idx)

X_train = X[train_idx]
X_test = X[test_idx]

def to_onehot(indices, num_classes):
    onehot = np.zeros((len(indices), num_classes))
    onehot[np.arange(len(indices)), indices] = 1
    return onehot

# Map mod labels to integers
mod_labels = [lbl[0] for lbl in labels]
Y_train = to_onehot([mods.index(mod_labels[i]) for i in train_idx], len(mods))
Y_test = to_onehot([mods.index(mod_labels[i]) for i in test_idx], len(mods))

# Input shape and class setup
in_shp = list(X_train.shape[1:])
classes = mods

# Build VT-CNN2 model
dropout_rate = 0.5
model = Sequential([
    Reshape([1] + in_shp, input_shape=in_shp),
    ZeroPadding2D((0, 2)),
    Conv2D(256, (1, 3), activation='relu', padding='valid', name='conv1', kernel_initializer='glorot_uniform'),
    Dropout(dropout_rate),
    ZeroPadding2D((0, 2)),
    Conv2D(80, (2, 3), activation='relu', padding='valid', name='conv2', kernel_initializer='glorot_uniform'),
    Dropout(dropout_rate),
    Flatten(),
    Dense(256, activation='relu', name='dense1', kernel_initializer='he_normal'),
    Dropout(dropout_rate),
    Dense(len(classes), kernel_initializer='he_normal', name='dense2'),
    Activation('softmax'),
    Reshape([len(classes)])
])

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

# Train the model
batch_size = 1024
epochs = 100
weights_path = 'convmodrecnets_CNN2_0.5.wts.h5'

history = model.fit(
    X_train, Y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
    validation_data=(X_test, Y_test),
    callbacks=[
        ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, mode='auto', verbose=0),
        EarlyStopping(monitor='val_loss', patience=5, mode='auto', verbose=0)
    ]
)

# Load the best model weights after training
model.load_weights(weights_path)
