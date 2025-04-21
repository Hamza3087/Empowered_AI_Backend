import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model("pakistani_currency_model.h5")

# Convert to SavedModel format (required for Hugging Face)
model.save("pakistani_currency_model.keras")
