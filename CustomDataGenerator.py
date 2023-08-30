from keras.preprocessing.image import ImageDataGenerator
from preprocessing_func import pre_processing
import random
import os
import cv2
import numpy as np

def get_data_gen_args(mode):

    """
    Generate ImageDataGenerator arguments (options) based on mode - (train, val, test)

    Arguments:
    mode -- which mode to use for ImageDataGenerator - 'train', 'test', 'val'

    Returns:
    ImageDataGenerator arguments for both input and the corresponding labels.

    """
    if mode == 'train' or mode == 'val':
        x_data_gen_args = dict(preprocessing_function=pre_processing,
                               horizontal_flip=True)

        y_data_gen_args = dict(horizontal_flip=True)

    elif mode == 'test':
        x_data_gen_args = dict(preprocessing_function=pre_processing)
        y_data_gen_args = dict()

    else:
        print("Data_generator function should get mode arg 'train' or 'val' or 'test'.")
        return -1

    return x_data_gen_args, y_data_gen_args

def data_generator(d_path, b_size, mode):

    """
    Implement Data Generator for Keras fit_generator.

    Arguments:
    d_path -- path of images and labels
    b_size -- batch size
    model -- 'train', 'val', 'test'

    Returns:
    yields batch_size of images during the learning process.

    """
    images = []
    masks = []

    image_dir = os.path.join(d_path, mode, "Image")
    mask_dir = os.path.join(d_path, mode, "Mask")

    for image_file in os.listdir(image_dir):
        if (image_file.lower().endswith(".png")):
            image_path = os.path.join(image_dir, image_file)

            # Generate mask file name based on image file name
            #mask_file = image_file.replace("_sub_image_", "_sub_mask_")  # Modify this as needed
            mask_file = image_file.replace("sub_image", "sub_mask").replace("H653A_", "H653A_3D_mask")
            print(mask_file)
            mask_path = os.path.join(mask_dir, mask_file)

            if os.path.exists(mask_path):
                image = cv2.imread(image_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                images.append(image)
                masks.append(mask)

    x_imgs = images
    y_imgs = masks

    # Make ImageDataGenerator.
    x_data_gen_args, y_data_gen_args = get_data_gen_args(mode)
    x_data_gen = ImageDataGenerator(**x_data_gen_args)
    y_data_gen = ImageDataGenerator(**y_data_gen_args)

    # random index for random data access.
    d_size = x_imgs[0].shape[0]
    shuffled_idx = list(range(d_size))

    x = []
    y = []
    while True:
        random.shuffle(shuffled_idx)
        for i in range(d_size):
            idx = shuffled_idx[i]

            x.append(x_imgs[idx].reshape((118, 178, 3)))
            y.append(y_imgs[idx].reshape((118, 178, 1)))

            if len(x) == b_size:
                # Adapt ImageDataGenerator flow method for data augmentation.
                _ = np.zeros(b_size)
                seed = random.randrange(1, 1000)

                x_tmp_gen = x_data_gen.flow(np.array(x), _,
                                            batch_size=b_size,
                                            seed=seed)
                y_tmp_gen = y_data_gen.flow(np.array(y), _,
                                            batch_size=b_size,
                                            seed=seed)

                # Finally, yield x, y data.
                x_result, _ = next(x_tmp_gen)
                y_result, _ = next(y_tmp_gen)

                yield x_result, y_result

                x.clear()
                y.clear()