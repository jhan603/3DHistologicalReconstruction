import numpy as np
import pandas as pd
import os
import cv2
import PIL.Image as Image
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from glob import glob
import random
from keras.models import Model
from keras.layers import Flatten, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator

Working_Directory = "D:\\Part4\\700AB\\3DHistologicalReconstruction\\H653A_11.3" # When working at home
#Working_Directory = " "                            # Root directory when working at university
Image_directory = Working_Directory + '\\Images'
os.makedirs(Image_directory, exist_ok=True)

Masks_directory = Working_Directory + '\\Mask'
os.makedirs(Masks_directory, exist_ok=True)

Has_Manual_Labels = ["0000", "0001", "0002", "0003", "0004", "0009", "0019", "0024" ,"0029"]
# "Only slices 1-5, 10, 15, 20, 25, 30 have manual labels at this stage" was said in the email but slide 0014 doesnt have the appropriate mask.
print("Each image/slide AND Mask has a size of 1424 by 2006 pixels\n")

def Img_Mask_Paths(Image_directory, Masks_directory, Has_Manual_Labels):
    """
    This function get the directory for each image and corresponding mask from the main Images and Masks directory.

    Loop through each path of image or mask and append it to the list of image/mask paths. We are working with 9 out of 200 ish images
    so we filter out the images that we will not be using. 

    **Args:**
        Image_directory (str): The directory path where the images are located.
        Masks_directory (str): The directory path where the masks are located.
        Has_Manual_Labels (list): A list of strings representing the labels used to filter the images and masks. Only the images/masks containing any of these labels will be included.

    **Returns:**
        image_path, mask_path: A tuple containing two lists. The first list contains the paths of the filtered images, and the second list contains the paths of the corresponding masks.
    """
    # a list to collect paths of images
    image_path = []
    for root, dirs, files in os.walk(Image_directory):
        # iterate over images
        for file in files:
            if (file.find(".png") != -1) & (file[0] == "H"):
                # create path
                path = os.path.join(root,file)
                # add path to list if it is one of the images we are using
                if any(label in file for label in Has_Manual_Labels):
                    image_path.append(path)
    

    # a list to collect paths masks
    mask_path = []
    for root, dirs, files in os.walk(Masks_directory):
        #iterate over masks
        for file in files:
            if (file.find(".png") != -1) & (file[0] == "H"):
                # obtain the path
                path = os.path.join(root,file)
                # add path to the list if it is one of the masks we are using
                if any(label in file for label in Has_Manual_Labels):
                    mask_path.append(path)
    
    #print("Number of image paths is", len(image_path))
    #print("Number of mask paths is", len(mask_path))

    return image_path, mask_path

def Split_Imgs_Masks(image_path, mask_path):
    # Define the maximum size for the sub-images
    # max_sub_size = 200  # Adjust this value if required
    # Made it not be in the loop anymore
    Split_Image_folder = Working_Directory + "\\Split_Images"
    os.makedirs(Split_Image_folder, exist_ok=True)
    
    # Create the destination folder if it doesn't exist
    Split_Mask_folder = Working_Directory +  "\\Split_Masks"
    os.makedirs(Split_Mask_folder, exist_ok=True)

    # paths = []
    for image, mask in zip(image_path, mask_path):
        image = Image.open(image)
        width, height = image.size
        #print(width, height)

        mask = Image.open(mask)
        #width, height = mask.size
        #print(width, height, "meowers")

        # Calculate the number of rows and columns in the grid
        num_rows = height // int(118) # 2006 divisible by 118 17 columns    
        #print(num_rows)
        num_cols = width //  int(178) # 1424 divisible by 178 8 rows
        #print(num_cols)

        # Calculate the actual size of the sub-images and sub-masks
        sub_width = width / num_cols
        #print(sub_width)
        sub_height = height / num_rows     
        #print(sub_height)   

        # Split the image into smaller images
        for row in range(num_rows):
            for col in range(num_cols):
                # Calculate the coordinates of the current sub-image
                left = col * sub_width
                upper = row * sub_height
                right = left + sub_width
                lower = upper + sub_height

                # Crop the sub-image from the original image
                sub_image = image.crop((left, upper, right, lower))
                sub_mask = mask.crop((left, upper, right, lower))

                # Get the original file name without the extension
                original_image_name = os.path.splitext(os.path.basename(image.filename))[0]
                original_mask_name = os.path.splitext(os.path.basename(mask.filename))[0]

                # Generate the new file name for the sub-image
                #sub_image_file_name = f"{original_image_name}_sub_image_{row}_{col}.png"
                #sub_mask_file_name = f"{original_mask_name}_sub_mask_{row}_{col}.png"
                sub_image_file_name = f"{original_image_name}_sub_image_{row:02d}_{col:02d}.png"
                sub_mask_file_name = f"{original_mask_name}_sub_mask_{row:02d}_{col:02d}.png"

                # Save the sub-image in the destination folder
                sub_image.save(os.path.join(Split_Image_folder, sub_image_file_name))
                sub_mask.save(os.path.join(Split_Mask_folder, sub_mask_file_name))
                # Append the image and mask paths as a pair to the list
                
                #paths.append((os.path.join(Split_Image_folder, sub_image_file_name), os.path.join(Split_Mask_folder, sub_mask_file_name)))  
    split_img_path, split_mask_path = Img_Mask_Paths(Split_Image_folder, Split_Mask_folder, Has_Manual_Labels)
    #split_img_path, split_mask_path = zip(*paths)
    return split_img_path, split_mask_path
    


def Decode_PNG_to_tensor(image_path, mask_path):
    """
    Splits the given images and masks into smaller sub-images and sub-masks based on a maximum size.

    Args:
        image_path (list): A list of file paths to the original images.
        mask_path (list): A list of file paths to the corresponding masks.

    Returns:
        tuple: A tuple containing two lists. The first list contains the paths of the split sub-images, and the second list contains the paths of the split sub-masks.
    """

    # create a list to store images
    images = []
    # iterate over image paths
    for path in (image_path):
        # read file
        file = tf.io.read_file(path)
        # decode png file into a tensor
        image = tf.image.decode_png(file, channels=3, dtype=tf.uint8)
        # append to the list
        images.append(image)
    
    # create a list to store masks
    masks = []
    # iterate over mask paths
    for path in (mask_path):
        # read the file
        file = tf.io.read_file(path)
        # decode png file into a tensor
        mask = tf.image.decode_png(file, channels=1, dtype=tf.uint16) # Multiple sources say to use 8-bit for greyscale image which is our mask technically
                                                                      #  but then mask breaks down if done like that
        # uint8: It is an 8-bit unsigned integer data type that can store values in the range of 0 to 255. It uses 1 byte (8 bits) of memory per pixel.
        # uint16: It is a 16-bit unsigned integer data type that can store values in the range of 0 to 65535. It uses 2 bytes (16 bits) of memory per pixel.

        # append mask to the list
        masks.append(mask)

    # Uncomment these two lines if needed
    # print("Number of images is", len(images))
    # print("Number of masks is", len(masks))

    return images, masks


def View_Image_Mask(split_images, split_masks, pic_num, split_img_path, split_mask_path):
    """
    Displays the specified image and mask from the given lists of split images and masks.

    Args:
        split_images (list): A list of split image tensors.
        split_masks (list): A list of split mask tensors.
        pic_num (int): The index of the image and mask to display.
        split_img_path (list): A list of file paths to the split images.
        split_mask_path (list): A list of file paths to the split masks.

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    try:
        image = split_images[pic_num]
        mask = split_masks[pic_num]

        split_image_name = split_img_path[pic_num]
        substring_image = split_image_name.split("\\")[-1].split(".")[0]

        split_mask_name = split_mask_path[pic_num]
        substring_mask = split_mask_name.split("\\")[-1].split(".")[0]

        image_size = mask.shape
        print("Image has a size of",image_size[0],"pixels by", image_size[1], "pixels")

        mask_size = image.shape
        print("Mask has a size of",mask_size[0],"pixels by", mask_size[1], "pixels")

        ax1.imshow(image)  # Display the image
        ax1.set_title("Image Plot of " + substring_image)

        ax2.imshow(mask, cmap='gray')  # Display the mask
        ax2.set_title("Mask Plot of " + substring_mask)
        
        plt.tight_layout()

        # Show the plot
        plt.show()
    except:
        print("SubImage / Sub Mask not found")



def is_image_all_black(image_path):
    # Open the image
    image = Image.open(image_path)

    # Convert the image to grayscale
    grayscale_image = image.convert("L")

    # Check if all pixels in the image are black (0)
    is_all_black = all(pixel == 0 for pixel in grayscale_image.getdata())

    return is_all_black

def remove_all_black(image_paths, mask_paths, split_images, split_masks):
    indices_to_remove = []
    
    for i, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        if is_image_all_black(image_path):
            indices_to_remove.append(i)
    
    # Remove the images, masks, and split images/masks tensors
    for index in sorted(indices_to_remove, reverse=True):
        del image_paths[index]
        del mask_paths[index]
        del split_images[index]
        del split_masks[index]
    
    return image_paths, mask_paths, split_images, split_masks

def No_Black_Split(split_img_paths, split_mask_paths):
    # Define the maximum size for the sub-images
    # max_sub_size = 200  # Adjust this value if required
    # Made it not be in the loop anymore
    No_Black_Split_Image_folder = Working_Directory + "\\No_Black_Images"
    os.makedirs(No_Black_Split_Image_folder, exist_ok=True)
    
    # Create the destination folder if it doesn't exist
    No_Black_Split_Mask_folder = Working_Directory +  "\\No_Black_Masks"
    os.makedirs(No_Black_Split_Mask_folder, exist_ok=True)

    # paths = []
    for sub_image_path, sub_mask_path in zip(split_img_paths, split_mask_paths):
        sub_image = Image.open(sub_image_path)
        sub_mask = Image.open(sub_mask_path)

        image_filename = os.path.basename(sub_image_path)
        mask_filename = os.path.basename(sub_mask_path)

        # Save the sub-image in the destination folder
        sub_image.save(os.path.join(No_Black_Split_Image_folder, image_filename))
        sub_mask.save(os.path.join(No_Black_Split_Mask_folder, mask_filename))

    image_paths = [file for file in os.listdir(No_Black_Split_Image_folder) if os.path.isfile(os.path.join(No_Black_Split_Image_folder, file))]
    image_paths.sort()
    mask_paths  = [file for file in os.listdir(No_Black_Split_Mask_folder) if os.path.isfile(os.path.join(No_Black_Split_Mask_folder, file))]
    mask_paths.sort()
    print("hi")
    return image_paths, mask_paths

def Split_Data(split_img_paths, split_mask_paths):
    num_samples = len(split_img_paths)
    indices = np.arange(num_samples)
    
    # Shuffle the indices
    seed = 512163833
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_images = []
    train_masks = []
    val_images = []
    val_masks = []
    test_images = []
    test_masks = []

    # Split the indices
    # Approx 70 15 15 split
    train_split = int(0.7 * num_samples)
    val_split = int(0.15 * num_samples)
    
    train_indices = indices[:train_split]
    val_indices = indices[train_split:(train_split + val_split)]
    test_indices = indices[(train_split + val_split):] 
    
    split_img_paths = np.array(split_img_paths)
    split_mask_paths = np.array(split_mask_paths)
    
    # Define folders for train, val, and test
    train_folder = Working_Directory + "\\Train"
    val_folder = Working_Directory + "\\Val"
    test_folder = Working_Directory + "\\Test"
    
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    # Define subfolders for images and masks within each train, val, and test folder
    train_img_folder = os.path.join(train_folder, "Image")
    train_mask_folder = os.path.join(train_folder, "Mask")
    val_img_folder = os.path.join(val_folder, "Image")
    val_mask_folder = os.path.join(val_folder, "Mask")
    test_img_folder = os.path.join(test_folder, "Image")
    test_mask_folder = os.path.join(test_folder, "Mask")
    
    os.makedirs(train_img_folder, exist_ok=True)
    os.makedirs(train_mask_folder, exist_ok=True)
    os.makedirs(val_img_folder, exist_ok=True)
    os.makedirs(val_mask_folder, exist_ok=True)
    os.makedirs(test_img_folder, exist_ok=True)
    os.makedirs(test_mask_folder, exist_ok=True)
    
    # Use the split indices to get the data
    train_img_paths = split_img_paths[train_indices]
    train_mask_paths = split_mask_paths[train_indices]
    
    val_img_paths = split_img_paths[val_indices]
    val_mask_paths = split_mask_paths[val_indices]
    
    test_img_paths = split_img_paths[test_indices]
    test_mask_paths = split_mask_paths[test_indices]

    # Loop through data and save to corresponding folders
    for img_path, mask_path in zip(train_img_paths, train_mask_paths):
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        img_name = os.path.basename(img_path)
        mask_name = os.path.basename(mask_path)

        img.save(os.path.join(train_img_folder, img_name))
        mask.save(os.path.join(train_mask_folder, mask_name))

        img_array = np.array(img)
        mask_array = np.array(mask)
        
        train_images.append(img_array)
        train_masks.append(mask_array)

    for img_path, mask_path in zip(val_img_paths, val_mask_paths):
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        img_name = os.path.basename(img_path)
        mask_name = os.path.basename(mask_path)

        img.save(os.path.join(val_img_folder, img_name))
        mask.save(os.path.join(val_mask_folder, mask_name))

        img_array = np.array(img)
        mask_array = np.array(mask)
        
        val_images.append(img_array)
        val_masks.append(mask_array)

    for img_path, mask_path in zip(test_img_paths, test_mask_paths):
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        img_name = os.path.basename(img_path)
        mask_name = os.path.basename(mask_path)

        img.save(os.path.join(test_img_folder, img_name))
        mask.save(os.path.join(test_mask_folder, mask_name))

        img_array = np.array(img)
        mask_array = np.array(mask)
        
        test_images.append(img_array)
        test_masks.append(mask_array)

    # Convert lists to numpy arrays
    train_images = np.array(train_images)
    train_masks = np.array(train_masks)
    val_images = np.array(val_images)
    val_masks = np.array(val_masks)
    test_images = np.array(test_images)
    test_masks = np.array(test_masks)
    
    return train_images, train_masks, val_images, val_masks, test_images, test_masks



if __name__ == "__main__":
    image_path, mask_path = Img_Mask_Paths(Image_directory, Masks_directory, Has_Manual_Labels)

    split_img_path, split_mask_path = Split_Imgs_Masks(image_path, mask_path)

    split_images, split_masks = Decode_PNG_to_tensor(split_img_path, split_mask_path)

    # View_Image_Mask(split_images, split_masks, pic_num=12, split_img_path = split_img_path, split_mask_path =  split_mask_path)

    split_img_paths, split_mask_paths, split_images, split_masks = remove_all_black(split_img_path, split_mask_path, split_images, split_masks)

    sub_image_paths, sub_mask_paths = No_Black_Split(split_img_paths, split_mask_paths)

    #train_img_path, train_mask_path, val_img_path, val_mask_path, test_img_path, test_mask_path = Split_data(split_img_paths, split_mask_paths, split_images, split_masks)
    train_images, train_masks, val_images, val_masks, test_images, test_masks = Split_Data(split_img_paths, split_mask_paths)

    # TrainModel(Working_Directory, "VGG16", layerNo = 1, epochs = 10, batch_size = 64, name = "VGG16")
    print("hello")