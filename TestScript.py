import numpy as np
import pandas as pd
import os
import PIL
import PIL.Image as Image
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import cv2
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from glob import glob

Working_Directory = "D:\\Part4\\700AB\\H653A_11.3" # When working at home
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
            if file.find(".png") != -1:
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

image_path, mask_path = Img_Mask_Paths(Image_directory, Masks_directory, Has_Manual_Labels)

# To Do: Split images, masks
# 1. Inputs are the image and mask paths (done)
# 2. Split the images and masks up but how i split an image i must split it up the same way with the masks
# 3. I need to name those images in a sensible and correct manner to understand. Eg: What Mask and perhaps what Index of split it is?
# idk that wording doesn't make sense but i get it. 
# Outputs: New Image, Mask Paths of split images

def Split_Imgs_Masks(image_path, mask_path):
    # Define the maximum size for the sub-images
    # max_sub_size = 200  # Adjust this value if required
    # Made it not be in the loop anymore
    Split_Image_folder = Working_Directory + "\\Split_Images"
    os.makedirs(Split_Image_folder, exist_ok=True)
    
    # Create the destination folder if it doesn't exist
    Split_Mask_folder = Working_Directory +  "\\Split_Masks"
    os.makedirs(Split_Mask_folder, exist_ok=True)

    for image, mask in zip(image_path, mask_path):
        image = Image.open(image)
        width, height = image.size
        #print(width, height)

        mask = Image.open(mask)
        #width, height = mask.size
        #print(width, height, "meowers")

        # Calculate the number of rows and columns in the grid
        num_rows = height // int(118) # 2006 divisible by 118 17 rows
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
                sub_image_file_name = f"{original_image_name}_sub_image_{row}_{col}.png"
                sub_mask_file_name = f"{original_mask_name}_sub_mask_{row}_{col}.png"

                # Save the sub-image in the destination folder
                sub_image.save(os.path.join(Split_Image_folder, sub_image_file_name))
                sub_mask.save(os.path.join(Split_Mask_folder, sub_mask_file_name))
        
    split_img_path, split_mask_path = Img_Mask_Paths(Split_Image_folder, Split_Mask_folder, Has_Manual_Labels)

    return split_img_path, split_mask_path
    
split_img_path, split_mask_path = Split_Imgs_Masks(image_path, mask_path)
#print(split_img_path, split_mask_path)

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

split_images, split_masks = Decode_PNG_to_tensor(split_img_path, split_mask_path)

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

View_Image_Mask(split_images, split_masks, pic_num=12, split_img_path = split_img_path, split_mask_path =  split_mask_path)

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

split_img_paths, split_mask_paths, split_images, split_masks = remove_all_black(split_img_path, split_mask_path, split_images, split_masks)

#print(split_img_paths)
print(len(split_img_paths))
print(len(split_mask_paths))
print(len(split_masks))

# 15th 16th July
# Changed the size of sub image and masks so that theyre size are whole values and no pixels are missed out
# Got rid of images and masks with corresponding tensors

# # Clearly its not plotting the mask wrong
        # Possible Reasons?
        # 1. the way im plotting is wrong? can't be cus if u view the tensor of an image its just all 0 values meaning all black
        # 2. The way im decoding the image is probably the reason why?
        # 3. Need to work on aligning the image and mask? Look closely at the example its not perfectly aligned? Try fix my splitting images method?
        # 4. Ask about uint8 and uint16 stuff
        # Questions to ask Alys
        # Will the person listening/marking be knowledgeable will we need to explain myo decid etc? just explain wat we hve done so far?
        # Good engineering knowledge and general biological knowledge, suggest couple of slides of the big picture problems and couple of slides 
        # little bit of anatomy, type of image processing image segemntation, focus on what my approach is, plans for remaining time. 
        # mid-year report, what should i include? is it individual report?
        # Show her my code and ask if my splits are fine?
        # when i view imgs using fiji why do u htink my masks are sometimes grey?
        # why does a mask look entirely black but when i view it in fiji it shows the mask fine? is it cus it lost its pixel labels? 
        # Only selecting subimages, with one type of tissue, figure out how to get rid of images and corresponding masks that have no information. 
        # Comments
        # Images weren't created in the same place, different types of origins?
        # Adjust spacing between subplots
        # image and a mask, blocks of tissues of 1 type of tissue.

# 1. Figure out why image and masks aren't aligning. By trying different separating, maybe add up images together STILL FIGURING OUT!
# 2. Figure out how to get rid of masks that provide no information DONE!
# 3. What about labels for images, 

# Firstly split the dataset into training and test set
# Rule of thumb a good split is 70 to 30
# Splitting Data into Training and Validation

def Split_Data(split_images, split_masks) :
    train_Image, val_Image,train_Mask, val_Mask = train_test_split(split_images, split_masks, test_size=0.3, 
                                                      random_state=512163833
                                                     )
    # develop tf Dataset objects
    train_X = tf.data.Dataset.from_tensor_slices(train_Image)
    val_X = tf.data.Dataset.from_tensor_slices(val_Image)

    train_y = tf.data.Dataset.from_tensor_slices(train_Mask)
    val_y = tf.data.Dataset.from_tensor_slices(val_Mask)

    # verify the shapes and data types
    train_X.element_spec, train_y.element_spec, val_X.element_spec, val_y.element_spec

    # zip images and masks
    train = tf.data.Dataset.zip((train_X, train_y))
    val = tf.data.Dataset.zip((val_X, val_y))
    return train, val

train_img_masks, val_img_masks = Split_Data(split_images, split_masks)
# Images_train, Images_test, Masks_train, Masks_test = train_test_split(images, masks, test_size=0.3, random_state=512163833)
print("hi")
