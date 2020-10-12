import os

import jsonpickle

try:
    from types import SimpleNamespace as Namespace
except ImportError:
    # Python 2.x fallback
    from argparse import Namespace


# Custom exception
class InvalidNNModelException(Exception):
    pass


# converts the whole model into script according to the template
def build(model):
    t_script = 'from tensorflow import keras\nimport jsonpickle\n\n\nif __name__ == \"__main__\":\n\twith open(\'generated\\\\data\', \'r\') as f:\n\t\tjShapedData = f.read()\n\tshaped_data = jsonpickle.decode(jShapedData)\n'
    for line in model.build_body():
        t_script += f'\t{line}\n'
    return t_script


# creates the script
def create_training_file(model):
    script = build(model)
    with open('generated\\training.py', 'w+') as f:
        f.write(script)

    try:
        import py_compile
        x = py_compile.compile("generated\\training.py", doraise=True)

    except py_compile.PyCompileError as e:
        raise InvalidNNModelException("Model has been created with mistakes, file doesn't compile properly. " + str(e))


# loads json model from a file
def load_json_model(path):
    with open(path, "r") as f:
        model = jsonpickle.decode(f.read())
    return model


# executes training of the NN
def exec_training_file():
    try:
        result = os.system('python generated//training.py')
    except Exception as e:
        print('Executable file doesn\'t exist. ' + str(e))
        return
    if 0 == result:
        print(" NN trained successfully")
    else:
        print(" NN didn't train successfully. There must be mistakes in the model.")


# loads model and converts it into script
def load_and_create():
    jmodel = None
    while jmodel is None:
        path = input("Please enter json model path: ")
        try:
            jmodel = load_json_model(path)
        except Exception as e:
            print("Json model path or json model is incorrect try again: " + str(e))

    try:
        create_training_file(jmodel)
    except InvalidNNModelException as e:
        print(str(e))


# creates script and executes training
def load_and_train():
    load_and_create()
    exec_training_file()


# shows options
def show_menu():
    print('NN Code Generator\n=======================\n')
    print('Choose what to do:')
    print('\t1) load model from json and generate code')
    print('\t2) run generated code and train model')
    print('\t3) load json model and run generated code, training the model (both previous options together)')
    print('\t4) exit')


# used to interact with the menu
def exec_menu():
    choice = 0
    while choice == 0:
        try:
            choice = int(input())
        except Exception as e:
            print('Enter a number! ' + str(e))

        if choice == 1:
            load_and_create()
        elif choice == 2:
            exec_training_file()
        elif choice == 3:
            load_and_train()
        elif choice == 4:
            return
        else:
            print('Number must be from 1 to 4.')
        show_menu()
        choice = 0


if __name__ == '__main__':
    show_menu()
    exec_menu()
