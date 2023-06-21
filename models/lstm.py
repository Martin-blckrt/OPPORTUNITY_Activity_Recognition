from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Bidirectional
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from utils import get_sequenced_data, plot_conf_mat

df_list, labels_list = get_sequenced_data()

cv_acc = []

# Loop over the data and labels
for i, (X_test, y_test) in enumerate(zip(df_list, labels_list)):
    X_train = np.asarray(np.concatenate([df for j, df in enumerate(df_list) if i != j])).astype('float32')
    y_train = np.array([element for j, sublist in enumerate(labels_list) if i != j for element in sublist]).reshape(-1, 1)

    X_test = np.asarray(X_test).astype('float32')
    y_test = np.array(y_test).reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    enc = enc.fit(y_train)

    y_train = enc.transform(y_train)
    y_test = enc.transform(y_test)

    model = Sequential()
    model.add(
        Bidirectional(
            LSTM(units=128, input_shape=[X_train.shape[1], X_train.shape[2]])
        )
    )
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['acc']
    )

    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        shuffle=False
    )

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    cv_acc.append(model.evaluate(X_test, y_test))

    y_pred = model.predict(X_test)

    plot_conf_mat(enc.inverse_transform(y_test), enc.inverse_transform(y_pred), enc.categories_[0])

print(cv_acc)
