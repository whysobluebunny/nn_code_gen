from tensorflow import keras
import jsonpickle


if __name__ == "__main__":
	with open('generated\\data', 'r') as f:
		jShapedData = f.read()
	shaped_data = jsonpickle.decode(jShapedData)
	model = keras.models.Sequential()
	model.add(keras.layers.Input( shape=(28, 28, 1)))
	model.add(keras.layers.Conv2D( use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',strides=(1, 1),activation='relu',filters=32,kernel_size=(3, 3),padding='valid',dilation_rate=(1, 1)))
	model.add(keras.layers.MaxPooling2D( pool_size=(2, 2)))
	model.add(keras.layers.Conv2D( use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',strides=(1, 1),activation='relu',filters=64,kernel_size=(3, 3),padding='valid',dilation_rate=(1, 1)))
	model.add(keras.layers.MaxPooling2D( pool_size=(2, 2)))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dropout( rate=0.5))
	model.add(keras.layers.Dense( use_bias=True,units=10,activation='softmax'))
	model.compile( optimizer='adam',loss='categorical_crossentropy')
	callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.003, patience=2)
	model.fit( x=shaped_data.x_training,y=shaped_data.y_training, batch_size=128,epochs=200,callbacks=[callback],validation_split=0.1)
	model.save('generated\\generated.h5')
