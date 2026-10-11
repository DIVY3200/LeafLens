"""Custom Keras layers used inside saved LeafLens models. Import before loading a model."""
import keras
from keras import ops


@keras.saving.register_keras_serializable(package="leaflens")
class CaffePreprocess(keras.layers.Layer):
    """VGG16's ImageNet input convention: RGB to BGR, then subtract the ImageNet channel means."""

    def call(self, inputs):
        mean = ops.convert_to_tensor([103.939, 116.779, 123.68], dtype=inputs.dtype)
        return ops.flip(inputs, axis=-1) - mean
