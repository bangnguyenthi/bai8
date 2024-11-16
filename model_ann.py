from tensorflow.keras import layers, models

def create_ann_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(150, 150, 3)),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
