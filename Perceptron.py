import tensorflow as tf
import numpy as np
import math
import os
import ImageLoader
import MNISTLoader


class Perceptron:
    INPUT_DIM = 784
    HIDDEN_DIM1 = 128
    HIDDEN_DIM2 = 64
    OUTPUT_DIM = 10

    EPOCH_NUM = 10
    LEARNING_RATE = 0.01
    BATCH_SIZE = 60

    if os.path.exists("./data/weights/perceptron_weights.npz"):
        print("Параметры инициализированы значениями из './data/weights/perceptron_weights.npz'")
        load_path = os.path.join("./data/weights", 'perceptron_weights.npz')
        load_data = np.load(load_path)
        W1 = tf.Variable(load_data['W1'])
        W2 = tf.Variable(load_data['W2'])
        W3 = tf.Variable(load_data['W3'])
        b1 = tf.Variable(load_data['b1'])
        b2 = tf.Variable(load_data['b2'])
        b3 = tf.Variable(load_data['b3'])
    else:
        print("Данные отсутствуют, инициализация параметров случаными значениями...")
        W1 = tf.Variable(np.random.randn(INPUT_DIM, HIDDEN_DIM1).astype(np.float32) * math.sqrt(2 / 784))
        W2 = tf.Variable(np.random.randn(HIDDEN_DIM1, HIDDEN_DIM2).astype(np.float32) * math.sqrt(2 / 128))
        W3 = tf.Variable(np.random.randn(HIDDEN_DIM2, OUTPUT_DIM).astype(np.float32) * math.sqrt(2 / 64))
        b1 = tf.Variable(np.zeros(128, dtype=np.float32))
        b2 = tf.Variable(np.zeros(64, dtype=np.float32))
        b3 = tf.Variable(np.zeros(10, dtype=np.float32))
    
    X = tf.expand_dims(ImageLoader.ImageLoader.load("./data/images/1.jpg"), axis=0)

    def inference(self):
        h1 = tf.nn.relu(self.X @ self.W1 + self.b1)
        h2 = tf.nn.relu(h1 @ self.W2 + self.b2)
        h3 = tf.nn.softmax(h2 @ self.W3 + self.b3)

        for index in range(len(h3[0])):
            value = h3[0][index]
            print("Вероятность числа", index, ": ", f"{value.numpy() * 100:.2f}%")
        

    def learn(self):
        mnist_data = MNISTLoader.MNISTLoader.load_data("./data/mnist_train.csv")
        x = mnist_data[0].astype(np.float32)/255.0
        y = MNISTLoader.MNISTLoader.one_hot_encode(mnist_data[1])

        indices = np.random.permutation(len(x))
        x_shuffled = tf.gather(x, indices)
        y_shuffled = tf.gather(y, indices)

        for epoch in range(self.EPOCH_NUM):
            print("Текущая эпоха: ", epoch + 1)           

            for i in range(0, len(x_shuffled), self.BATCH_SIZE):
                batch_x = x_shuffled[i:i+self.BATCH_SIZE]
                batch_y = y_shuffled[i:i+self.BATCH_SIZE]

                t1 = batch_x @ self.W1 + self.b1
                h1 = tf.nn.relu(t1)
                t2 = h1 @ self.W2 + self.b2
                h2 = tf.nn.relu(t2)
                t3 = h2 @ self.W3 + self.b3
                h3 = tf.nn.softmax(t3)

                dE_dt3 = h3 - batch_y
                dE_dh2 = dE_dt3 @ tf.transpose(self.W3)
                dE_dt2 = dE_dh2 * tf.cast(t2 > 0, tf.float32)
                dE_dh1 = dE_dt2 @ tf.transpose(self.W2)
                dE_dt1 = dE_dh1 * tf.cast(t1 > 0, tf.float32)

                dE_dW3 = tf.transpose(h2) @ dE_dt3 / self.BATCH_SIZE
                dE_dW2 = tf.transpose(h1) @ dE_dt2 / self.BATCH_SIZE
                dE_dW1 = tf.transpose(batch_x) @ dE_dt1 / self.BATCH_SIZE
                dE_db3 = tf.reduce_mean(dE_dt3, axis=0) 
                dE_db2 = tf.reduce_mean(dE_dt2, axis=0) 
                dE_db1 = tf.reduce_mean(dE_dt1, axis=0) 

                self.W1 = self.W1 - self.LEARNING_RATE * dE_dW1
                self.W2 = self.W2 - self.LEARNING_RATE * dE_dW2
                self.W3 = self.W3 - self.LEARNING_RATE * dE_dW3
                self.b1 = self.b1 - self.LEARNING_RATE * dE_db1
                self.b2 = self.b2 - self.LEARNING_RATE * dE_db2
                self.b3 = self.b3 - self.LEARNING_RATE * dE_db3

        self.save_weights(self ,'./data/weights')

    def save_weights(self ,folder_path):
        save_path = os.path.join(folder_path, 'perceptron_weights.npz')

        np.savez(save_path,
            W1 = self.W1.numpy(),
            W2 = self.W2.numpy(),
            W3 = self.W3.numpy(),
            b1=self.b1.numpy(),
            b2=self.b2.numpy(),
            b3=self.b3.numpy()
        )


perc = Perceptron
# perc.learn(perc)
perc.inference(perc)
