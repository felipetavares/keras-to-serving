#!/usr/bin/env python3

# Loads a given keras h5 file with weights and exports
# ready for tf-serving

import tensorflow as tf
from keras.models import load_model

import argparse

# Command line arguments
parser = argparse.ArgumentParser(description='Loads a given keras h5 file with weights and exports ready for tf-serving')
parser.add_argument('model', help='The .h5 keras model')
parser.add_argument('weights', help='The .h5 keras weights')
parser.add_argument('destination', help='Destination folder')

args = parser.parse_args()

# Load the model
model = load_model(args.model)
model.load_weights(args.weights)

# Create a tf session
session = tf.Session()
session.run(tf.global_variables_initializer())

# Save the model
builder = tf.saved_model.builder.SavedModelBuilder(args.destination)
builder.add_meta_graph_and_variables(
      session, [tf.saved_model.tag_constants.SERVING],
      main_op=tf.tables_initializer(),
      strip_default_attrs=True)
builder.save()
