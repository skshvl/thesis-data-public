
from tqdm import tqdm
from PIL import Image
import numpy as np
import os, copy, json

import cv2
from matplotlib import pyplot as plt


import shap


N_ROWS = 4

background_mask = np.zeros((N_ROWS, N_ROWS))

def vector_to_masked_images(masking_matrixes, original_image, blurred_image, N_ROWS):
    """Apply masking matrix to image
    masking_matrix should be N_ROWS, N_ROWS
    
    original_image should be 224 x 224 x 3
    
    value of 1 -> that patch is kept
    value of 0 -> that patch is blurred"""

    results = []

    patch_size = 224 // N_ROWS

    for masking_matrix in masking_matrixes:

        result = copy.deepcopy(original_image)

        for row in range(N_ROWS):
            for col in range(N_ROWS):
                #  boundaries of the current patch
                start_row, start_col = row * patch_size, col * patch_size
                end_row, end_col = start_row + patch_size, start_col + patch_size
                
                # If the mask value is 0, apply the blurred value
                if masking_matrix[row, col] == 0:
                    result[start_row:end_row, start_col:end_col] = \
                        blurred_image[start_row:end_row, start_col:end_col]

        results.append(result)
    
    return results

def blur_image(image, blur_value = 99):
    """Blurs numpy image"""
    blurred_image = cv2.GaussianBlur(image, (blur_value, blur_value), 0)
    return blurred_image


global model


def generate_shap(img_path, caption, foil, prediction_function, model, model_name = "unknown_model", verbose = False, plots = True, export_plots = False, return_variant_count = False):
    """Generate SHAP given a row of dataframe

    prediction_function should take images(list), caption, foil

    verbose = 1: basic
    verbose = 2: graph iamges

    if return_variant_count == True, return the number of image variants evaluated alongside the final SHAP map
    
    """
    # keep track of how many total images + image variants we are evaluating in the run of SHAP on a single stimulus

    total_images_evaluated = 0

    image = Image.open(img_path).convert('RGB')
    image = np.array(image.resize((224,224)))
    blurred_image = blur_image(image) # this is the background

    if verbose: print("Image shape:", image.shape)



    def local_predict(masking_matrixes):

        nonlocal total_images_evaluated
        total_images_evaluated += len(masking_matrixes) # len(masking_matrixes) is how many variants of the image this run is predicting on


        global model

        """Define prediction funciton that can works with variable created above (within generate_shap function!

        Turn single image into a series of masked images and run prediction function on each

        Given images=[image], make prediction involving image and caption and foil"""


        # take masked_matrixes and turn it into a list of masked images. Masked means substituting the blurred background
        images = vector_to_masked_images(masking_matrixes, # this variablew varies per shap masking instance
                                         image, blurred_image,  # these two are fixed per image
                                         N_ROWS)
        
        predictions = prediction_function(images, caption, foil)
        
        if verbose >= 2:
            print(masking_matrixes)
            for index, img in enumerate(images):
                print("Prediction for following image is", predictions[index])
                plt.imshow(img)
                plt.show()
        
        return predictions

    # create SHAP explainer
    # masker = shap.maskers.Image("inpaint_telea", (224, 224, 3))
    masker = shap.maskers.Image(np.array([background_mask])) # background mask is of size (4,4): ZEROES
    explainer = shap.Explainer(local_predict, masker, 
                               seed = 42)

    # get SHAP values
    starting_matrix = np.array([np.arange(1, N_ROWS**2+1).reshape(N_ROWS, N_ROWS)]) # initialize shap matrix to turn individual cells on and off
    # this matrix goes like [1 2 3 4] [5 6 7 8 ] [9 10 11 12] [13 14 15 16] -- if any are set to 0 that patch will be blurred
    if verbose: print("SHAP matrix shape:", starting_matrix.shape)
    shap_values = explainer(starting_matrix, max_evals=300, batch_size=64,) # GENERATE THE ACTUAL SHAP VALUES

    # generate graphs if showing or exporting plots is true
    if plots or export_plots: 
        plt.figure()
        plt.title(f"SHAP attribution map for {model}")
        shap.plots.image(shap_values=shap_values.values[0][..., np.newaxis],
                        pixel_values=image, show = False)

        # Add the caption
        plt.figtext(0.5, 0.02, f"Caption: {caption}, \nFoil: {foil}", wrap=True, horizontalalignment='center', fontsize=7)

        # Save the plot as a PNG file
        if export_plots: plt.savefig(f"xai_images/{os.path.basename(img_path)}_{model_name}_SHAP.png", bbox_inches='tight', pad_inches=0.5)
        if plots: plt.show()
        plt.close()
    
    


    if return_variant_count:
        if verbose: print(f"Calculated SHAP values on {os.path.basename(img_path)} by evaluating a total of {total_images_evaluated} image variants.")
        return shap_values.values[0], total_images_evaluated

    return shap_values.values[0]