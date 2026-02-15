import tensorflow as tf
from tensorflow.keras.applications import ResNet50, DenseNet201
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model


def build_model(base_arch_name, input_shape=(256, 256, 3), num_classes=2):
    input_tensor = Input(shape=input_shape)

    if base_arch_name == "ResNet50":
        base_model = ResNet50(weights="imagenet",
                              include_top=False,
                              input_tensor=input_tensor)
    elif base_arch_name == "DenseNet201":
        base_model = DenseNet201(weights="imagenet",
                                 include_top=False,
                                 input_tensor=input_tensor)
    else:
        raise ValueError("Unsupported architecture")

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    return model


def weighted_ensemble(model1, model2, w1=0.5, w2=0.5):
    input_layer = Input(shape=(256, 256, 3))

    out1 = model1(input_layer)
    out2 = model2(input_layer)

    weighted_output = w1 * out1 + w2 * out2

    ensemble_model = Model(inputs=input_layer, outputs=weighted_output)

    return ensemble_model
