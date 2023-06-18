
import tensorflow as tf
import matplotlib.cm as cm
import numpy as np

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def heat_map(img, model_predict,sousmodel,model_masked, last_conv_layer_name,resize = None):

  img_to_process = np.expand_dims(img, axis=0)
  img_to_process = model_masked.predict(img_to_process,verbose = 0)

  # Obtenir les prédictions et les couches de sortie du modèle
  preds = model_predict.predict(img_to_process,verbose = 0)
  pred_class = np.argmax(preds[0])
  if resize is None :
    heatmap = make_gradcam_heatmap(img_to_process, sousmodel,last_conv_layer_name , pred_index=pred_class)
  else:
    resize =  tf.image.resize(img_to_process[0],(resize,resize))
    img_to_process = tf.expand_dims(resize, axis=0)
    heatmap = make_gradcam_heatmap(img_to_process, sousmodel,last_conv_layer_name , pred_index=pred_class)

  # Rescale heatmap to a range 0-255
  heatmap = np.uint8(255 * heatmap)

  # Use jet colormap to colorize heatmap
  jet = cm.get_cmap("coolwarm")

  # Use RGB values of the colormap
  jet_colors = jet(np.arange(256))[:, :3]
  jet_heatmap = jet_colors[heatmap]

  # Create an image with RGB colorized heatmap
  jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
  jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
  jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

  # Superimpose the heatmap on original image
  superimposed_img = jet_heatmap * .9 + img
  superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

  # Display Grad CAM
  return superimposed_img


def grad_cam_fonction(img,model,model_tl_index, last_layer_name,resize=None):
  masked,model =model.layers[0],model.layers[1]
  model_tl = model.layers[model_tl_index]
  return heat_map(img,model,model_tl,masked,last_layer_name,resize)