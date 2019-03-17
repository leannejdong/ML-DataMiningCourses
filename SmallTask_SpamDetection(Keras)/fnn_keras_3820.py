
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2


def main():

    training_data = np.loadtxt("train_data.csv", delimiter=',')  # Iris classification problem (UCI dataset)
    test_data = np.loadtxt("test_data.csv", delimiter=',')  #
    hidden_size = 2
    input_size = 57
    output_size = 1
    x_train = training_data[:, 0:input_size]
    y_train = training_data[:, input_size:input_size+output_size]
    x_test = test_data[:, 0:input_size]
    y_test = test_data[:, input_size:input_size+output_size]


    

    model = Sequential()
    model.add(Dense(hidden_size, activation='relu', input_shape=(input_size,), kernel_regularizer=l2(0.1)))
    #model.add(Dense(hidden_size, activation='relu'))
    #model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(output_size, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, batch_size=5, verbose=2, epochs=50)
    eval = model.evaluate(x_test, y_test)

    print("\nModel Loss: "+str(eval[0]))
    print("Model Accuracy: "+str(eval[1]))

    # Predicting the Test set results
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5)

    # Creating the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

if __name__ == "__main__": main()
