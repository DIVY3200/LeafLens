import json
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Data Preprocessing
training_set = tf.keras.utils.image_dataset_from_directory(
    'Dataset/old/train',
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
    verbose=True,
)

# validation Image Processing
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
    verbose=True,
)

# print(training_set)

# to avoid overshooting 1. Choose small learning rate default 0.001 we are taking 0.0001 2. There may be a chance of
# underfitting, so increase the no. of neurons 3. Add more convolution layer to extract more feature from image there
# may be possibility that model unable to capture relevant features or model is confusing due to lack of feature
for x, y in training_set:
    print(x, x.shape)
    print(y, y.shape)
    break

# Building Model
cnn = tf.keras.models.Sequential()

cnn.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(128, 128, 3)))
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=2, strides=2))


cnn.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
cnn.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=2, strides=2))


cnn.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
cnn.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=2, strides=2))


cnn.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
cnn.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=2, strides=2))

cnn.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
cnn.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Dropout(0.25))

cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=1500, activation='relu'))

cnn.add(tf.keras.layers.Dropout(0.4))

cnn.add(tf.keras.layers.Dense(units=38, activation='softmax'))

cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy',
            metrics=['accuracy'])

cnn.summary()

training_history = cnn.fit(x=training_set, validation_data=validation_set, epochs=10)

# Model Evaluation
train_loss, train_acc = cnn.evaluate(training_set)
print(train_loss, train_acc)

val_loss, val_acc = cnn.evaluate(validation_set)
print(val_loss, val_acc)

# Saving Model
cnn.save('trained_plant_disease_model.keras')  # compress my model

# training_history.history

# Recording History in json
with open("training_hist.json", "w") as f:
    json.dump(training_history.history, f)

print(training_history.history.keys())

# Accuracy Visualization
epochs = [i for i in range(1, 11)]
# print(epochs)
plt.plot(epochs, training_history.history['accuracy'], color='red', label='Training Accuracy')
# plt.show()
plt.plot(epochs, training_history.history['val_accuracy'], color='blue', label='Validation Accuracy')
# plt.show()
plt.xlabel("No. of Epochs")
plt.ylabel("Accuracy Result")
plt.title("Visualization of Accuracy Result")
plt.legend()
plt.show()

# Some other metrics for model evaluation
class_name = validation_set.class_names
# print(class_name)
test_set = tf.keras.utils.image_dataset_from_directory(
    'Dataset/old/valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=1,
    image_size=(128, 128),
    shuffle=False,  # Shuffle become false
    seed=None,
    validation_split=None,
    subset=None,
    follow_links=False,
    crop_to_aspect_ratio=False
)

# Perform Prediction
y_pred = cnn.predict(test_set)
# print(y_pred, y_pred.shape)
predicted_categories = tf.argmax(y_pred, axis=1)  # perform in column wise
# print(predicted_categories)

true_categories = tf.concat([y for x, y in test_set], axis=0)  # In row-wise
# print(true_categories)

Y_true = tf.argmax(true_categories, axis=1)
# print(Y_true)

# predicted_categories
cm = confusion_matrix(Y_true, predicted_categories)
# cm.shape();

print(classification_report(Y_true, predicted_categories, target_names=class_name))


# Confusion Matrix Visualization
plt.figure(figsize=(40, 40))
sns.heatmap(cm, annot=True, annot_kws={"size": 10})
plt.xlabel('Predicted Class', fontsize=20)
plt.ylabel('Actual Class', fontsize=20)
plt.title('Plant Disease Prediction Confusion Matrix', fontsize=25)
plt.show()
