#!/usr/bin/env python3

# Loads a given keras h5 file with weights and exports
# ready for tf-serving

import keras.backend as K
import tensorflow as tf
from keras.models import load_model

import argparse

# Command line arguments
parser = argparse.ArgumentParser(description='Loads a given keras h5 file with weights and exports ready for tf-serving')
parser.add_argument('model', help='The .h5 keras model')
parser.add_argument('weights', help='The .h5 keras weights')
parser.add_argument('destination', help='Destination folder')
parser.add_argument('--summary', action='store_true', help='Prints model summary and exits')

args = parser.parse_args()

# Create a tf session
session = tf.Session()
session.run(tf.global_variables_initializer())

K.set_session(session)

# Load the model
model = load_model(args.model)
model.load_weights(args.weights)

if args.summary:
  # Prints out summary
  model.summary()
else:
  # Create the SignatureDef
  inputs = model.inputs[0]
  outputs = model.outputs[0]

  tensor_info_inputs = tf.saved_model.utils.build_tensor_info(inputs) 
  tensor_info_outputs = tf.saved_model.utils.build_tensor_info(outputs)

  signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
      inputs={inputs.name: tensor_info_inputs},
      outputs={outputs.name: tensor_info_outputs},
      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )
  )

  # Save the model
  try:
    builder = tf.saved_model.builder.SavedModelBuilder(args.destination)
    builder.add_meta_graph_and_variables(
          session, [tf.saved_model.tag_constants.SERVING],
          main_op=tf.tables_initializer(),
          signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
          })

    builder.save()
  except AssertionError as e:
    print(e)
