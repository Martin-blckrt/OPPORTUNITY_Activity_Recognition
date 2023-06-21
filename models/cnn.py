import numpy as np
from tqdm import tqdm
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Dropout, concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from utils import get_sequenced_data, plot_line
import statistics

# fit and evaluate a model
def evaluate_multi_headed_model(trainX, trainy, testX, testy, n):
    filter_num, kernel_size = 16, 3
    verbose, epochs, batch_size = 0, n, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    # head 1
    inputs1 = Input(shape=(n_timesteps, n_features))
    conv1 = Conv1D(filters=filter_num, kernel_size=3, activation='relu')(inputs1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)

    # head 2
    inputs2 = Input(shape=(n_timesteps, n_features))
    conv2 = Conv1D(filters=filter_num, kernel_size=5, activation='relu')(inputs2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)

    # head 3
    inputs3 = Input(shape=(n_timesteps, n_features))
    conv3 = Conv1D(filters=filter_num, kernel_size=11, activation='relu')(inputs3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)

    # merge
    merged = concatenate([flat1, flat2, flat3])

    # interpretation
    dense1 = Dense(100, activation='relu')(merged)
    outputs = Dense(n_outputs, activation='softmax')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit([trainX, trainX, trainX], trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # evaluate model
    _, accuracy = model.evaluate([testX, testX, testX], testy, batch_size=batch_size, verbose=0)

    # plot_conf_mat(enc.inverse_transform(np.argmax(testy, axis=1)), enc.inverse_transform(model.predict(np.argmax(testy, axis=1))), enc.categories_[0])

    return accuracy


# fit and evaluate a model
def evaluate_1d_model(trainX, trainy, testX, testy, n):
    verbose, epochs, batch_size = 0, n, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=4, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=4, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy


df_list, labels_list = get_sequenced_data()
acc_rate = []

# Loop over different numbers of epochs
for n in tqdm([10, 20, 50, 100, 200]):
    cv_acc = []
    # Cross-validation loop
    for i, (X_test, y_test) in enumerate(zip(df_list, labels_list)):
        # Prepare training data by excluding the current fold
        X_train = np.asarray(np.concatenate([df for j, df in enumerate(df_list) if i != j])).astype('float32')
        y_train = np.array([element for j, sublist in enumerate(labels_list) if i != j for element in sublist]).ravel()

        X_test = np.asarray(X_test).astype('float32')
        y_test = np.array(y_test).ravel()

        # Encode labels
        enc = LabelEncoder().fit(y_train)
        y_train = enc.transform(y_train)
        y_test = enc.transform(y_test)

        # One-hot encode y
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # Evaluate the 1D model and collect accuracy
        cv_acc.append(evaluate_1d_model(X_train, y_train, X_test, y_test, n))

    print(statistics.mean(cv_acc))
    acc_rate.append(statistics.mean(cv_acc))

# Plot the accuracy evolution
plot_line(acc_rate, [10, 20, 50, 100, 200], 'Evolution of accuracy with respect to CNN epochs', 'Number of epochs', 'Accuracy')
