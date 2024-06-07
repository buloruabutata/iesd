import tensorflow as tf
from tensorflow.keras import layers, models

def AFNet():
    model = models.Sequential([
        layers.Conv2D(filters=3, kernel_size=(8, 1), strides=(2, 1), padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Conv2D(filters=5, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Conv2D(filters=7, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Conv2D(filters=9, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Conv2D(filters=16, kernel_size=(4, 1), strides=(1, 1), padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Flatten(),  # 将卷积层输出扁平化处理，以便输入到全连接层
        layers.Dropout(0.5),
        layers.Dense(16, activation='softmax'),
        layers.Dense(2)
    ])

    return model
