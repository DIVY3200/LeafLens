"""LeafLens: plant disease recognition.

Submodules:
  classifier  loads a trained model (.keras with TensorFlow, .tflite with LiteRT) and classifies photos
  knowledge   symptoms, treatment and prevention for every class
  datasets    PlantVillage + FieldPlant loading, label mapping and leak-free splits
  layers      custom Keras layers that saved models may contain
"""
