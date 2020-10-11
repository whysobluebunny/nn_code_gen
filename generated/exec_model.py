import jsonpickle
import numpy
from tensorflow import keras


def load_json_model(path):
    with open(path, 'r') as f:
        loaded_data = jsonpickle.decode(f.read())
    return loaded_data


def get_data():
    loaded_data = None
    while loaded_data is None:
        path = input('Please enter data file path: ')
        try:
            loaded_data = load_json_model(path)
        except Exception as e:
            print('Path is incorrect try again: ' + str(e))
    return loaded_data


def show_menu():
    print('Choose which field to use in NN: ')
    print('\t1) x_training')
    print('\t2) y_training')
    print('\t3) x_test')
    print('\t4) y_test')


def show_file_menu():
    print('Where do you want to put the predicted data?')
    print('1) Console')
    print('2) File')


def write_file(data):
    while (True):
        path = input('Please enter data file path: ')
        try:
            numpy.savetxt(path, data, delimiter=',')
            break
        except Exception as e:
            print('Path is incorrect try again: ' + str(e))


if __name__ == '__main__':
    while (True):
        data = get_data()

        try:
            model = keras.models.load_model('generated.h5')
            with open('data', 'r') as f:
                jShapedData = f.read()
            shaped_data = jsonpickle.decode(jShapedData)
        except Exception as e:
            print('Trained model file or data file doesn\'t exist: ' + str(e))

        show_menu()
        choice = 0
        while choice == 0:
            try:
                choice = int(input())
            except Exception as e:
                print('Enter a number! ' + str(e))

            if choice == 1:
                data_field = shaped_data.x_training
            elif choice == 2:
                data_field = shaped_data.y_training
            elif choice == 3:
                data_field = shaped_data.x_test
            elif choice == 4:
                data_field = shaped_data.y_test
            else:
                print('Number must be from 1 to 4.')
                show_menu()
                choice = 0

        prediction = model.predict(data_field)

        show_file_menu()
        choice = 0
        while choice == 0:
            try:
                choice = int(input())
            except Exception as e:
                print('Enter a number! ' + str(e))

            if choice == 1:
                for pred in prediction:
                    print(pred)
            elif choice == 2:
                write_file(prediction)
            else:
                print('Number must be from 1 to 2.')
                show_file_menu()
                choice = 0

        again = input('Do you want to continue? Type \'yes\' to continue').lower()
        if again != 'yes':
            break
