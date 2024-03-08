from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, make_response
import numpy as np
from PIL import Image

import random


APP_DIRECTORY = "." # the prefix to put before "/img" and other subdirectories of the app folder

import io

from utils import * # lots of variables imported from utils
from data_structures import AppDataHolder # import AppDataHolder class

# non GUI matplotlib backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


app = Flask(__name__)

# constant variables are declared in utils, changing variables here


stimulus_idx = 0 # initialize to stimulus 0
global app_data
app_data = AppDataHolder(initial_stimulus_index = 0, stimuli_dataframe = STIMULI_DF) # initialize and reset data 

@app.route('/')
def intro():
    return render_template('index.html')


@app.route('/next')
def next():
    """Load next available stimulus (based on current sitmulus index)"""

    success = app_data.load_next_stimulus() # populate app_data with attributes of next stimulus

    if success == True: # if new stimulus was available, load it
        print("Loading new stimulus")

        # add HTML to caption foil highlighting difference
        caption, foil = highlight_differences(app_data.caption, app_data.foil)

        # shuffle order of caption, foil into caption1 and caption2
        # also shuffle the labels accordingly, so the form knows which is caption which is foil
        [caption1, cap1_label], [caption2, cap2_label] = shuffle_caption_foil(caption,foil)

        return render_template('task.html', 
                               show_mask = SHOW_MASK, 
                               caption1 = caption1, 
                               cap1_label = cap1_label,
                               caption2 = caption2, 
                               cap2_label = cap2_label,
                               orange_threshold = CLICK_THRESHOLDS[0],
                            red_threshold = CLICK_THRESHOLDS[1])
    # if FINISHED
    else: 
        # if finished, export the data as DATAFRAME about this user interaction
        pd.DataFrame.from_dict(app_data.output_dict).to_pickle(os.path.join("../app_output", app_data.instance_id+".pickle"))
        return intro()


@app.route('/get_image')
def get_image():
    """Send current image to website as jpeg"""

    image = Image.fromarray(app_data.current_image.astype('uint8'))
    
    # this BytesIO will hold the image
    buf = io.BytesIO()
    
    # save image as jpeg in the buf object
    image.save(buf, format='JPEG')
    
    # rewind  buffer's file pointer
    buf.seek(0)
    
    return send_file(buf, mimetype="image/jpeg")

@app.route('/get_mask')
def get_mask():
    """Send a mask visualization plot as jpeg to the website"""

    # Create a figure. Considering 100 dpi, for 200 pixels width we want width=2 inches
    fig, ax = plt.subplots(figsize=(2, 2))

    downsampled_mask = app_data.current_mask[::7, ::7]

    # Display heatmap
    cax = ax.imshow(downsampled_mask, cmap='viridis', vmin=INITIAL_MASK_VALUE, vmax=MAX_MASK_VALUE)

    # Add a colorbar
    plt.colorbar(cax)
    
    # Save figure
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', dpi=200) # save to buf data structure
    buf.seek(0)
    
    return send_file(buf, mimetype="image/jpeg")

@app.route('/select_caption', methods=['POST'])
def record_caption_choice():
    """This receives information about what the user selected a the caption"""
    data = request.get_json()
    selection = data.get('selection')

    # process the selection 
    print(f'User selected: {selection}')

    app_data.answer = selection

    return jsonify({'status': 'ok'})

@app.route('/deblur', methods=['POST'])
def deblur():
    """Deblur the image and save the new image as current_image, also save the updated mask"""

    data = request.json
    x = data['x']
    y = data['y']

    # calculate new mask and image
    mask_delta = deblur_util(x, y,  app_data.current_mask) # diff between old mask and new mask
    # get updated image and updated mask (only changes pixels affected by mask_delta)
    app_data.current_image, app_data.current_mask = update_image_and_mask(app_data.current_image,
                                                        app_data.current_mask,
                                                        mask_delta,
                                                        app_data.blur_level_images,
                                                        verbose = False)
    
    app_data.points_clicked.append((x,y)) # add x, y to record of points clicked on this image by this user

    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True)