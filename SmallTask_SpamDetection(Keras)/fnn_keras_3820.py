import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix


def main():

    training_data = np.loadtxt("train_data.csv", delimiter=',')  # Iris classification problem (UCI dataset)
    test_data = np.loadtxt("test_data.csv", delimiter=',')  #
    hidden_size = 2
    input_size = 58
    output_size = 1
    x_train = training_data[:, 0:input_size]
    y_train = training_data[:, input_size:input_size+output_size]
    x_test = test_data[:, 0:input_size]
    y_test = test_data[:, input_size:input_size+output_size]


    # Model definition
    model1 = Sequential()
    model1.add(Dense(hidden_size, activation='relu', input_shape=(input_size,), kernel_regularizer=l2(0.1)))
    #model.add(Dense(hidden_size, activation='relu'))
    #model.add(Dense(hidden_size, activation='relu'))
    model1.add(Dense(output_size, activation='sigmoid'))
    model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model1.summary()
    # fit model
    model1.fit(x_train, y_train, batch_size=32, verbose=2, epochs=500)
    # Model Evaluation
    eval = model1.evaluate(x_test, y_test)

    print("\nModel Loss: "+str(eval[0]))
    print("Model Accuracy: "+str(eval[1]))

    # Predicting the Test set results
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5)

    # Creating the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

 #   Model Loss: 0.05364268836767777
#Model Accuracy: 0.9920289855072464
#[[812   9]
# [  2 557]]


if __name__ == "__main__": main()
