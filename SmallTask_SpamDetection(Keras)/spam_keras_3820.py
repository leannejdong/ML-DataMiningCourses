import pandas as pd
import numpy as np
#np.random.seed(123) # for reproducibility
from keras.models import Sequential  # building NN layer by layer
from keras.layers import Dense, Dropout # randomly initialise weights to small number close to 0
from keras.regularizers import l2 # reduce the risk of overfitting
from sklearn.metrics import confusion_matrix

def main():
	problem = 1 # [1,2,3] choose your problem (Iris classfication or 4-bit parity or XOR gate)
        

	if problem == 1:
		training_data = pd.read_csv("train_data.csv")  # Iris classification problem (UCI dataset)
		test_data = pd.read_csv("test_data.csv")  #
		hidden_size = 2
		input_size = 58
		output_size = 1

		x_train = training_data.iloc[:,0:input_size]
		y_train = training_data.iloc[:,input_size:input_size + output_size]
		x_test = test_data.iloc[:, 0:input_size]
		y_test = test_data.iloc[:, input_size:input_size + output_size]
		
	if problem == 2:
		training_data = pd.read_csv("train.csv")  # Iris classification problem (UCI dataset)
		test_data = pd.read_csv("test.csv")  #
		hidden_size = 2
		input_size = 4
		output_size = 2

		x_train = training_data.iloc[:,0:input_size]
		y_train = training_data.iloc[:,input_size:input_size + output_size]
		x_test = test_data.iloc[:, 0:input_size]
		y_test = test_data.iloc[:, input_size:input_size + output_size]

	    # Model definition
	model1 = Sequential() # no need to put any parameter as we are defining layer manually
	model1.add(Dense(hidden_size, activation='relu', input_shape=(input_size,), kernel_regularizer=l2(0.1)))
	model1.add(Dropout(0.5))
	#model.add(Dense(2, activation='relu', input_shape=(58,)))
	#model.add(Dense(6, activation='relu'))
	#model.add(Dense(6, activation='relu'))
	model1.add(Dense(output_size, activation='sigmoid'))
	model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	model1.summary()
    # fit model
	history = model1.fit(x_train, y_train, batch_size=32, verbose=1, epochs=500)
    # Model Evaluation
	eval = model1.evaluate(x_test, y_test)
	print("\nModel Loss: "+str(eval[0]))
	print("Model Accuracy: "+str(eval[1]))

    # Summarise history for accuracy
    #pyplot.plot(history.history['acc'], label='train')
    #pyplot.plot(history.history['val_acc'], label='test')
    #pyplot.legend()
    #pyplot.show()
    
    # Predicting the Test set results. 
    #If the output is > 0.5, then label as spam else no spam
	y_pred = model1.predict(x_test)
	y_pred = (y_pred > 0.5)
       
    # Creating the Confusion Matrix
	cm = confusion_matrix(y_test, y_pred)
	print(cm)

if __name__ == "__main__": main()

#Model Loss: 0.04245432259608128
#Model Accuracy: 0.9825960841189267
#[[820   1]
# [ 23 535]]

