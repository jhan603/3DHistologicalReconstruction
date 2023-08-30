import numpy as np

def pre_processing(img):

    """
    Pre-preprocessing of the input image.

    Arguments:
    img -- input image

    Returns:
    returns the normalized image having values between -1 to +1

    """

    # Centering helps normalization image (-1 ~ 1 value)
    return img / 255