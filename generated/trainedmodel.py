from tensorflow import keras
from keras.models import load_model

if __name__ == "__main__":
    actionType = input("Enter data file name or type \'exit\' to exit:").lower()
    while actionType != 'exit':
        try:
            model = load_model('generated.h5')
            with open('data', 'r') as f:
                shData = f.read()
        except Exception as e:
            print("Failed downloading file from file system: " + str(e))
        try:
            # TODO
            model.predict()
        except Exception as e:
            print() #TODO вывод информации дополнительно в файл.