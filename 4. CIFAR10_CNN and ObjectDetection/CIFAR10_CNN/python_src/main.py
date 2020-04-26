import keras
from keras.datasets import cifar10

from CNN import CNN, TimingCallback()

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    num_classes = 10
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    structure = [
        ('conv',32),
        ('conv',32),
        ('pool',2),
        ('conv',64),
        ('conv',64),
        ('pool',2),
        ('dense',512)
    ]
    call_back = TimingCallback()
    cifar10_cnn = CNN(x_train,y_train,x_test,y_test)
    cifar10_cnn.set_architecture(structure,batch_normalization=True,activation='relu',learning_optimizer='rmsprop')
    cifar10_cnn.compile(batch_size=100,epochs=100,call_back=call_back)
