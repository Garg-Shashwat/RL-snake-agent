import tensorflow as tf
print('1')
model_path = "model_B_30"
new_model = tf.keras.models.load_model(model_path)
#model = tf.saved_model.load(model_path)
print('2')
print(new_model.summary())
new_model.save('MODEL-1_60.h5')