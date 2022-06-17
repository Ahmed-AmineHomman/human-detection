# Welcome to my Human Detection project ! #

This project is my first attempt at playing with Computer Vision models for detecting people (faces, bodies) in images
or videos. It contains two scripts: *image.py* and *video.py* that are respectively designed for detecting people in
images and videos. These scripts will take an image or a video respectively as input and produce an annotated version of
this input with boxes where people are detected.

## Getting Started ##

### Prerequisites

This code requires the following libraries:

| name       | tested version |
| ----       | -------------- |
| tensorflow |  2.4.1         |
| opencv     |  4.5.1         |
| imutils    |  0.5.4         |
| numpy      |  1.20.2        |

**Remark**:
the code might work with older versions of the above libraries, but it has not been tested with them.

**Remark on tensorflow**:
the code has been written in order to work with tensorflow version 2.4.1, but it solely uses methods inside the module
*tensorflow.compat.v1*. Therefore, if you have the first version of tensorflow installed in your system (1.5 or above),
you could replace all occurrences of "tf.compat.v1" inside the file "detectors/RCNNDetector.py" and the code should
work.

### Quickstart

The executable scripts of this repository are the "image.py" and "video.py" scripts. When calling them, you should
provide a configuration file as first argument, like so:

```
python image.py path_to_my_config_file
```

```
python video.py path_to_my_config_file
```

where *python* represents the path to your python executable (in unix systems writing only *python* should work) and
*path_to_my_config_file* is the path to your configuration file.

Two examples of configuration files (one for each script) with my optimal parameters are available in the
"config_example" folder.
This repository is ignoring any folder name "config" in its root directory.
Therefore, here is the procedure to follow for a quick usage of the scripts:

1. create a 'config' directory in the root folder where you cloned this repository in your system,
2. copy both configuration files in the 'config_examples' folder in your newly created 'config' folder,
3. change the values of the 'output_folder' argument to the folder of your choice where the scripts will write the
annotated image/video,
4. run the script of your choice by providing its associated configuration with one of the following command lines:
```
python image.py config/image.json
```
```
python video.py config/video.json
```

### Configuration File

The configuration file must be a json-formatted file and contain at least the following arguments:

|      key name |   type | value                                                                             |
| ------------- | ------ | --------------------------------------------------------------------------------- |
| capture_path  | string | the path to your image or video                                                   |
| output_folder | string | the path to the folder where you want the code to write the annotated image/video |

Additional arguments can also be provided:

|    key name |       type |                      default value | value |
| ----------- | ---------- | ---------------------------------- | ----- |
| resolution  | list       | []                                 | resolution of the outputted image/video (**necessary for video.py**) |
| thresholds  | dictionary | {"confidence": 0, "overlap": 0.3}  | thresholds parameters for the model (see "Parametrization" section for details)|
| logging     | dictionary | {"display": true, "console": true} | activates display of result image/video and logging into console |
| frames      | dictionary | {"min": 0, "max": -1}              | defines min/max frames at which detection is performed (only for video.py), set "max" to -1 for no maximum frames |
| box_color   | list       | \[0, 255, 0\]                      | RGB color of the annotated bounding boxes |

**Remark**: when "display" logging is activated with "video.py", you can press 'q' at any time to stop the execution.
Any key will work (and close the displayed image) when running "image.py" however.

## Inputs
Some standard inputs are available in the 'input' folder if you want to play with the scripts:

| name          |                                                                                          description |
| ------------- | ---------------------------------------------------------------------------------------------------- |
| portraits.jpg | an image containing a grid of portraits of people with different background color                    |
| bodies.jpg    | an image containing a grid of people upper bodies in white background                                |
| video.mp4     | a video containing the Dior commercial [available here](https://www.youtube.com/watch?v=h4s0llOpKrU) |

# Modelization ##

In the remaining of this document, we will call the models detecting people *detectors*.
All the detectors are implemented in the 'detectors' subpackage.
They all implement the same API, defined in the *AbstractHumanDetector* class (in the
'detectors/AbstractHumanDetector.py' file).

In both "image.py" and "video.py", the detector must be defined by the *detector* variable.
If you wish to use a different model than the default one, you must define *detector* with the class of your choice
among the ones provided in the 'detectors' subpackage.
The fact that all classes implement the same API ensures the correct execution of the code whichever class you choose to
use.

Overlapping boxes outputted by detectors are then merged using *non-max suppression* (implemented in *imutils* package
[here](https://github.com/jrosebr1/imutils/blob/master/imutils/object_detection.py)).

## Default detector
Both scripts uses a RCNN Detection model to detect people.
This model is taken from the *TensorFlow Detection Model Zoo* (for tensorflow version 1), available
[here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md), and the
model used is the *faster_rcnn_inception_v2_coco*.

The code requires the model graph ('.pb' file) in order to work, available in the 'data' folder.
The scripts will automatically search for the appropriate file in this folder, so you have nothing to do except make sure
that the 'data' folder contains the '.pb' file (that should be included in this deposit).

## Additional models
Additional models are available in the 'detectors' subpackage.
They all have been tested but work less optimally than the default model.

### Haar Cascade Classifiers (HCC)
Detectors using opencv *Haar Cascade Classifiers* (see this [opencv tutorial](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html#face-detection))
are available in the 'detectors' subpackage, with the *HaarCascadeDetector" class (implemented in 
"detectors/HaarCascadeDetector.py").
This implementation needs the xml file of the corresponding opencv HCC.
Files corresponding to all opencv's trained detectors can be found [in this url](https://github.com/opencv/opencv/tree/master/data/haarcascades).
However, we have included some of them (the most appropriate ones) in the 'data' folder of this repository.

**Remark**:
The code has been tested with the detectors corresponding to the xml files in this repository, but not with the others,
so correct execution is not guaranteed when using any other file than those available in this deposit.

### Histogram of Oriented Gradients (HOG)
*Histogram of Oriented Gradients* (see the [wikipedia article](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients))
have also been implemented and tested, and are available in the 'detectors' subpackage with the *HOGDetector* class
(implemented in "detectors/HOGDetector.py").
This detector needs no file in order to work.

**Remark**:
The HOG detectors do not perform well on the tested inputs, producing many false positives (i.e. boxes where no people
can be found).

## Parametrization
They are two common parameters accepted by all detectors that can be defined in the configuration file:

- *confidence threshold*: minimal confidence of the boxes outputted by the detector.
  Each detector can produce a score along with the detected bounding box.
  This score represents the confidence of the model in the fact that the object detected is indeed a human.
  Specifying a high confidence threshold will cause the models to output only boxes for which they are certain that they
  detected a human.
  **Remark**: the magnitude of the confidence is not the same for all models (RCNNs will produce probabilities (i.e.
  confidences between 0 and 1) where HCCs typically produce confidence levels that can scale up to 10).
- *overlap threshold*: overlapping threshold above which boxes are merged.
  This is a parameter of the *non-max suppression*: overlaps of each box pairs are computed, and pairs whose overlap is
  greater than this threshold are merged together.
  This parameter takes values between 0 and 1, and values between 0.3 and 0.5 are typically used.