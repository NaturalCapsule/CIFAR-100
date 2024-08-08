# Importing libraries and layers
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets.cifar100 import load_data
import numpy as np

# loading the datasets (cifar-100)
(x_train, y_train), (x_test, y_test) = load_data()

# getting class names for this dataset
classes = [
    "beaver", "dolphin", "otter", "seal", "whale",
    "aquarium fish", "flatfish", "ray", "shark", "trout",
    "orchids", "poppies", "roses", "sunflowers", "tulips",
    "bottles", "bowls", "cans", "cups", "plates",
    "apples", "mushrooms", "oranges", "pears", "sweet peppers",
    "clock", "computer keyboard", "lamp", "telephone", "television",
    "bed", "chair", "couch", "table", "wardrobe",
    "bee", "beetle", "butterfly", "caterpillar", "cockroach",
    "bear", "leopard", "lion", "tiger", "wolf",
    "bridge", "castle", "house", "road", "skyscraper",
    "cloud", "forest", "mountain", "plain", "sea",
    "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
    "fox", "porcupine", "possum", "raccoon", "skunk",
    "crab", "lobster", "snail", "spider", "worm",
    "baby", "boy", "girl", "man", "woman",
    "crocodile", "dinosaur", "lizard", "snake", "turtle",
    "hamster", "mouse", "rabbit", "shrew", "squirrel",
    "maple", "oak", "palm", "pine", "willow",
    "bicycle", "bus", "motorcycle", "pickup truck", "train",
    "lawn-mower", "rocket", "streetcar", "tank", "tractor"
]

# making the x_train and x_test float32 data type
x_train = np.array(x_train).astype('float32')
x_test = np.array(x_test).astype('float32')

# Normalizing features
x_train /= 255.0
x_test /= 255.0

# Building a data augemtation layers for the model
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
])

#Builing the model with the data augmentation layers in it
model = tf.keras.Sequential([
    data_augmentation,
    layers.Input(shape=(32, 32, 3)),

    layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2, 2), strides = 2),
    layers.Dropout(0.2),

    layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2, 2), strides = 2),
    layers.Dropout(0.3),

    layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2, 2), strides = 2),
    layers.Dropout(0.3),

    layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2, 2), strides = 2),
    layers.Dropout(0.3),

    layers.Flatten(),

    layers.Dense(units=2000, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(units=500, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(units=100, activation='softmax')
])

# Creating callbacks for the model
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 0.9**epoch)
early_stopping = tf.keras.callbacks.EarlyStopping(patience = 5, verbose = 2)

# Compiling and fitting the model
model.compile(optimizer = tf.keras.optimizers.Adam(), loss = tf.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])
model_history = model.fit(x = x_train, y = y_train, validation_data = (x_test, y_test), callbacks = [early_stopping, lr_schedule], epochs = 100)