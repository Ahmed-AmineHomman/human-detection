import cv2


def parse_arguments(configuration):
    """
    Parses arguments from dictionary file.

    This method returns a dictionary containing values for all arguments.
    Default values are applied when the input dictionary does not contain a key corresponding to the argument.
    :param configuration: dictionary containing arguments.
    :return: dictionary containing (key, value) couples for all arguments.
    """

    def _assign_dict(main_key, values):
        if main_key not in configuration.keys():
            output = values
        elif not isinstance(configuration[main_key], dict):
            output = values
        else:
            output = dict()
            for label, value in values.items():
                if label not in configuration[main_key].keys():
                    output[label] = value
                else:
                    output[label] = configuration[main_key][label]
        return output

    def _assign_list(main_key, values):
        if main_key not in configuration.keys():
            output = values
        elif not isinstance(configuration[main_key], list):
            output = values
        else:
            output = configuration[main_key]
        return output

    # initialize output
    arguments = dict()

    # mandatory parameters
    for key in ["capture_path", "output_folder"]:
        if key not in configuration.keys():
            raise Exception(f"you must provide '{key}' argument")
        else:
            arguments[key] = configuration[key]

    # logging parameters
    arguments["logging"] = _assign_dict(main_key="logging", values={"console": True, "display": True})

    # resolution
    arguments["resolution"] = _assign_list(main_key="resolution", values=())
    arguments["resolution"] = tuple(arguments["resolution"])  # opencv needs tuple, not list

    # min/max frames
    arguments["frames"] = _assign_dict(main_key="frames", values={"min": 0, "max": -1})

    # bounding boxes color
    arguments["box_color"] = _assign_list(main_key="box_color", values=[0, 255, 0])

    # thresholds (confidence for bounding boxes and overlap for non-max suppression)
    arguments["thresholds"] = _assign_dict(main_key="thresholds", values={"confidence": 0, "overlap": 0.3})

    return arguments


def add_box(frame, box, text=None, color=(0, 255, 0), thickness=1):
    """
    Adds the specified bounding box to the specified frame.

    This methods adds a bounding box to a given frame, and returns the annotated frame.
    The user can also specify additional parameters of the box, such as its color or its thickness.
    A label can be also added to the box if a text is given to the method.

    The box must be specified as an iterable of four floats.
    The first two represents the coordinates of the upper left corner of the box and the second two represent
    respectively its width and heigth.

    If provided, the label will be placed in the top left corner of the bounding box.

    :param frame: np.ndarray representing the image to annotate,
    :param box: iterable containing four floats representing the upper left coordinates of the box a
    :param text:
    :param color:
    :param thickness:
    :return:
    """
    x, y, w, h = box
    cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=color, thickness=thickness)

    if text is not None:
        x_text = x + 5
        y_text = y + 20 if y + 20 < frame.shape[1] - 15 else y - 15
        cv2.putText(img=frame, text=text, org=(x_text, y_text), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75, color=color, thickness=thickness)

    return frame
