import lenet
import matplotlib.pyplot as plt

class epoch_50 (lenet.my_model):
    def __init__(self, batch_size, num_classes, epochs, learning_rate, img_rows, img_cols):
        super().__init__(batch_size, num_classes, epochs, learning_rate, img_rows, img_cols)
    
    def train(self):
        self.history = self.model.fit(model.x_train, model.y_train,
                    batch_size=model.batch_size,
                    epochs=model.epochs,
                    verbose=1,
                    validation_data=(model.x_test, model.y_test)
                    )
        self.model.save_weights('my_model_50.h5')
    def plot(self):      
        plt.figure()
        plt.title('Accuracy')
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc= 'lower right')

        plt.figure()
        plt.title('Loss')
        plt.plot(self.history.history['loss'])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')

        plt.show()



if __name__ == '__main__':
    model = epoch_50(32,10,50,0.001,28,28)
    model.load()
    model.build()
    model.train()
    model.plot()

    