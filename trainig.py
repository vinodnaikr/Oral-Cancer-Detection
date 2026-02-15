import tensorflow as tf


def compile_and_train(model, train_gen, val_gen, epochs=20):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs
    )

    return history
