import tensorflow as tf


def mobile_v(input_shape, weights=None):
    model = tf.keras.applications.MobileNetV2(
        weights=weights,
        input_shape=input_shape,
        input_tensor=None,
        include_top=False,
        pooling=None,
    )

    # Make flatten layer for bounding box and label heads
    input_layer = model.output
    flatten_layer = tf.keras.layers.Flatten()(input_layer)

    # The head which output 4 bounding box values
    bbox_head = tf.keras.layers.Dense(128, activation="relu")(flatten_layer)
    bbox_head = tf.keras.layers.Dense(64, activation="relu")(bbox_head)
    bbox_head = tf.keras.layers.Dense(32, activation="relu")(bbox_head)
    bbox_head = tf.keras.layers.Dense(4, activation="sigmoid", name="bounding_box")(bbox_head)

    # The head which output label name for detected object
    label_head = tf.keras.layers.Dense(512, activation="relu")(flatten_layer)
    label_head = tf.keras.layers.Dense(256, activation="relu")(label_head)
    label_head = tf.keras.layers.Dropout(0.2)(label_head)
    label_head = tf.keras.layers.Dense(200, activation="softmax", name="class_label")(label_head)

    # putting the model together
    model = tf.keras.models.Model(inputs=model.input, outputs=(bbox_head, label_head))

    loss_functions = {
        "class_label": "sparse_categorical_crossentropy",
        "bounding_box": "mean_squared_error",
    }
    loss_weights = {
        "class_label": 1.0,
        "bounding_box": 1.0
    }
    optimizer = tf.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss=loss_functions, loss_weights=loss_weights, optimizer=optimizer, metrics=["accuracy"])

    return model
