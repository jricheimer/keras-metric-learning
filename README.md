# Keras Metric Learning Library
Deep Metric Learning Library for Keras

## Welcome
The Keras Metric Learning Library provides the Keras-user with the functionality
to train models with the metric-learning losses being published in the research
literature. See this post for more info.

## Getting Started
Go ahead and clone the repo
```
git clone http://github.com/jricheimer/keras-metric-learning
```
If you'd like to experiment with the Stanford Online Products dataset (a nice
size dataset made for testing metric learning approaches), take a minute to
download it here.

Then, use our script to generate the hdf5 files for the dataset:
```
cd keras-metric-learning
mkdir dataset && cd dataset
python kml_create_stanford_hdf5.py --root_path /path/to/stanford/dataset
```

After that's finished processing, you should have two hdf5 files (one for train,
one for test) in the dataset directory.

## Start training

Take a look at the example notebooks to see how to use the library functionalities.
Enjoy!