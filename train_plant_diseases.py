import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

validation_set = tf.keras.utils.image_dataset_from_directory(
    'Dataset/old/valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    follow_links=False,
    crop_to_aspect_ratio=False
)
class_name = validation_set.class_names
print(class_name)

# Loading Model
cnn = tf.keras.models.load_model('trained_plant_disease_model.keras')
# model.summary()

# Visualizing Single Image of Test set
image_path = 'Dataset/old/test/CornCommonRust2.JPG'

# Reading an image
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR image into RGB

# Displaying Image
plt.imshow(img)
plt.title('Test Image')
plt.xticks([])  # Remove X-axis
plt.yticks([])  # Remove Y-axis
plt.show()

# Testing Model
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
# print(input_arr.shape)
input_arr = np.array([input_arr])  # Convert single image to a batch.
# print(input_arr.shape)
predictions = cnn.predict(input_arr)
# print(predictions, predictions.shape)

result_index = np.argmax(predictions)
print(result_index)

# Displaying the disease prediction
model_prediction = class_name[result_index]
plt.imshow(img)
plt.title(f"Disease Name: {model_prediction}")
plt.xticks([])
plt.yticks([])
plt.show()
