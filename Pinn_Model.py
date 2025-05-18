import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        # Define the layers
        self.hidden_layers = [tf.keras.layers.Dense(100, activation='tanh') for _ in range(4)]
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
    
    def get_config(self):
        # Return the configuration of the model as a dictionary
        config = super(PINN, self).get_config()  # Get the base model config
        return config
    
    @classmethod
    def from_config(cls, config):
        # Rebuild the model from the configuration (optional but recommended)
        return cls()
