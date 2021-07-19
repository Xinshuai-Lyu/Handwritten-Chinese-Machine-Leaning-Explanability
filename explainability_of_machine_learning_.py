#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Code Author: C2038737


# In[111]:


from PIL import Image

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', '')
import matplotlib.gridspec as gridspec
from matplotlib.text import Text
from matplotlib.widgets import Button

import os


# ### Data preparation 

# #### models all pixels influential power (by coefficients)

# In[112]:


def get_linear_model(input_size):
    input1 = keras.layers.Input(shape=(input_size*input_size))
    output = keras.layers.Dense(1, activation='sigmoid')(input1)
    linear_model = keras.Model(inputs=[input1], outputs=[output])
    linear_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return linear_model
# 8x8
linear_model = get_linear_model(8)
checkpoint_path = "xinshuai_models/linearmodel/8\\0243.ckpt"
linear_model.load_weights(checkpoint_path)
_8 = list(np.reshape(linear_model.variables[0].numpy(), 8*8))
# 16x16
linear_model = get_linear_model(16)
checkpoint_path = "xinshuai_models/linearmodel/16\\0171.ckpt"
linear_model.load_weights(checkpoint_path)
_16 = list(np.reshape(linear_model.variables[0].numpy(), 16*16))
# 32x32
linear_model = get_linear_model(32)
checkpoint_path = "xinshuai_models/linearmodel/32\\0062.ckpt"
linear_model.load_weights(checkpoint_path)
_32 = list(np.reshape(linear_model.variables[0].numpy(), 32*32))
# 64x64
linear_model = get_linear_model(64)
checkpoint_path = "xinshuai_models/linearmodel/64\\0054.ckpt"
linear_model.load_weights(checkpoint_path)
_64 = list(np.reshape(linear_model.variables[0].numpy(), 64*64))
# 128x128
linear_model = get_linear_model(128)
checkpoint_path = "xinshuai_models/linearmodel/128\\0026.ckpt"
linear_model.load_weights(checkpoint_path)
_128 = list(np.reshape(linear_model.variables[0].numpy(), 128*128))

models_pixels_influential_power_by_coefficients = [_8, _16, _32, _64, _128]


# #### models input sizes and accuracy results

# In[113]:


df = pd.read_csv("accuracy_of_input_size")
input_sizes = df["input_sizes"].to_list()
accuracys = df["accuracys"].to_list()


# #### 火(Fire) and 水(water) images

# In[114]:


# base path
train_base_path = os.path.join(os.curdir, "Train")

# 火(Fire) path
fire = "火"
fire_base_path = os.path.join(train_base_path, fire)

# 水(Water) path
water = "水"
water_base_path = os.path.join(train_base_path, water)

fires = os.listdir(fire_base_path)
waters = os.listdir(water_base_path)


# In[115]:


def get_random_fire_water():
    fire_index = int(np.random.random() * len(fires))
    water_index = int(np.random.random() * len(waters))
    fire_image = Image.open(os.path.join(fire_base_path, fires[fire_index]))
    water_image = Image.open(os.path.join(water_base_path, waters[water_index]))
    # fire and water images
    fire_image = fire_image.convert('L')
    water_image = water_image.convert('L')
    return fire_image, water_image


# #### models top 10 influential power pixels (by coefficients)

# In[116]:


def get_Nth_influential_power_pixel_coefficient(model_coefficients, N):
    top_N = model_coefficients[:N]
    top_N.sort()
    top_N = list(top_N)
    i = N
    while i < len(model_coefficients):
        top_N.append(model_coefficients[i])
        top_N.sort()
        top_N.pop(0)
        i += 1
    return top_N[0]

def get_top_N_influential_power_pixels(which_model, N):
    model_coefficients = models_pixels_influential_power_by_coefficients[which_model]
    model_input_size = input_sizes[which_model]
    the_Nth_influential_power_pixel_coefficient = get_Nth_influential_power_pixel_coefficient(model_coefficients, N)
    model_pixels_y_position = []
    model_pixels_x_position = []
    i = len(model_coefficients) - 1
    for model_coefficient in model_coefficients:
        if model_coefficient >= the_Nth_influential_power_pixel_coefficient:
            model_pixels_y_position.append(i // model_input_size)
            model_pixels_x_position.append((model_input_size-1) - i % model_input_size)
        i -= 1
    return model_pixels_x_position, model_pixels_y_position, model_input_size


# In[117]:


best_model = np.argmax(accuracys)
N = 20
best_model_x, best_model_y, best_model_input_size = get_top_N_influential_power_pixels(best_model, N)

_8model_x, _8model_y, _8model_input_size = get_top_N_influential_power_pixels(0, int(8*8*0.05))
_16model_x, _16model_y, _16model_input_size = get_top_N_influential_power_pixels(1, int(16*16*0.05))
_32model_x, _32model_y, _32model_input_size = get_top_N_influential_power_pixels(2, N)
_64model_x, _64model_y, _64model_input_size = get_top_N_influential_power_pixels(3, N)
_128model_x, _128model_y, _128model_input_size = get_top_N_influential_power_pixels(4, N)

xs = [_8model_x, _16model_x, _32model_x, _64model_x, _128model_x]
ys = [_8model_y, _16model_y, _32model_y, _64model_y, _128model_y]
model_input_sizes = [_8model_input_size, _16model_input_size, _32model_input_size, 
                     _64model_input_size, _128model_input_size]


# ### Data Interactive Visualization

# #### Style graphs

# In[118]:


def format_axes(axes, is_image=False):
    for ax in axes:
        ax.tick_params(labelsize="xx-small", pad=-2)
        ax.tick_params(top=False, left=False, bottom=False, right=False)
        if is_image:
            ax.tick_params(labeltop=False, labelleft=False, labelbottom=False, labelright=False)
            ax.axis('off')
def set_main_title(text):
    plt.suptitle(text, size="x-small", weight="bold")
def set_sub_title(ax, text):
    ax.set_title(text, size="xx-small", pad=2, weight="bold")
def set_label(ax, text, flag="x"):
    if flag == "x":
        ax.set_xlabel(text, size="xx-small")
    else:
        ax.set_ylabel(text, size="xx-small")
def text_markers(ax, model_x, model_y, top_N_influential_power_pixels):
    i = 0
    for coefficient in top_N_influential_power_pixels:
        ax.text(model_x[i],
                 model_y[i], 
                 str(coefficient), 
                 size="xx-small")
        i += 1


# #### Contain graphs in Grid

# In[119]:


def create_grid():
    fig = plt.figure()
    gs0 = gridspec.GridSpec(1, 2, figure=fig)
    gs00 = gridspec.GridSpecFromSubplotSpec(5, 6, subplot_spec=gs0[0])
    ax1 = fig.add_subplot(gs00[0:1, 0:3])
    ax2 = fig.add_subplot(gs00[0:1, 3:6])
    ax3 = fig.add_subplot(gs00[1:3, :])
    ax4 = fig.add_subplot(gs00[3:5, :])
    gs01 = gridspec.GridSpecFromSubplotSpec(5, 6, subplot_spec=gs0[1])
    ax_button = fig.add_subplot(gs01[0:1, 1:5])
    ax5 = fig.add_subplot(gs01[1:3, :])
    ax6 = fig.add_subplot(gs01[3:5, :])
    return fig, ax1, ax2, ax3, ax4, ax_button, ax5, ax6


# #### Paint graphs and Make graphs interactive

# In[120]:


fire_image, water_image = get_random_fire_water()
resized_fires = [fire_image.resize((8,8)), fire_image.resize((16,16)),
             fire_image.resize((32,32)), fire_image.resize((64,64)),
             fire_image.resize((128,128))]
resized_waters = [water_image.resize((8,8)), water_image.resize((16,16)),
                 water_image.resize((32,32)), water_image.resize((64,64)),
                 water_image.resize((128,128))]


# In[121]:


def draw_ax1_ax2(fire_image, water_image):
    ax1.figure.canvas.draw_idle()
    ax1.cla()
    ax2.figure.canvas.draw_idle()
    ax2.cla()
    set_sub_title(ax1, "Chinese Handwritten Fire")
    set_sub_title(ax2, "Water")
    ax1.imshow(fire_image, cmap="gray")
    ax2.imshow(water_image, cmap="gray")
    format_axes([ax1, ax2], is_image=True)
def draw_ax3(model_index=-1):
    colors = ['r', 'r', 'r', 'r', 'r']
    ax3.figure.canvas.draw_idle()
    ax3.cla()
    colors[model_index] = 'green'
    set_sub_title(ax3, "Accuracy as function of input size in linear model")
    set_label(ax3, "input size", "x")
    set_label(ax3, "accuracy", "y")
    ax3.scatter(input_sizes, accuracys, marker="8", picker=True, 
                    c=['r', 'r', 'r', 'r', 'r'])
    ax3.legend(labels=["UNSELECTED"], labelcolor=["red"],
                    loc="lower right", title="STATUS", fontsize="xx-small", title_fontsize="xx-small")
    ax3.scatter(input_sizes, accuracys, marker="8", picker=True, 
                    c=colors)
    ax3.plot(input_sizes, accuracys)
    format_axes([ax3])
def draw_ax4(model_index):
    ax4.cla()
    ax4.figure.canvas.draw_idle()
    model_x = xs[model_index]
    model_y = ys[model_index]
    model_input_size = model_input_sizes[model_index]
    set_sub_title(ax4, f"The pixels model({model_input_size}x{model_input_size}) focuses on")
    ax4.set_xlim(0, model_input_size)
    ax4.set_ylim(0, model_input_size)
    ax4.scatter(model_x, model_y, marker="8", color="red")
    set_label(ax4, "x", "x")
    set_label(ax4, "y", "y")
    format_axes([ax4])
def draw_image(ax, image, model_index):
    image_input_size = model_input_sizes[model_index]
    image = np.array(image)
    image = np.reshape(image, image_input_size*image_input_size)
    model_pixels_y_position = []
    model_pixels_x_position = []
    i = len(image) - 1
    for pixel in image:
        if pixel < 255:
            model_pixels_y_position.append(i // image_input_size)
            model_pixels_x_position.append((image_input_size-1) - (i % image_input_size))
        i -= 1
    ax.scatter(model_pixels_x_position, model_pixels_y_position, s=1, color="black")
def draw_ax5_ax6(model_index):
    ax5.cla()
    ax5.figure.canvas.draw_idle()
    ax6.cla()
    ax6.figure.canvas.draw_idle()
    model_x = xs[model_index]
    model_y = ys[model_index]
    model_input_size = model_input_sizes[model_index]
    ax5.scatter(model_x, model_y, marker="8", color="red")
    ax5.set_xlim(0, model_input_size)
    ax5.set_ylim(0, model_input_size)#
    draw_image(ax5, resized_fires[model_index], model_index)

    ax6.scatter(model_x, model_y, marker="8", color="red")
    ax6.set_xlim(0, model_input_size)
    ax6.set_ylim(0, model_input_size)
    draw_image(ax6, resized_waters[model_index], model_index)
    
    set_sub_title(ax5, f"The pixels model({model_input_size}x{model_input_size}) focuses on")
    set_label(ax5, "x", "x")
    set_label(ax5, "y", "y")
    set_label(ax6, "x", "x")
    set_label(ax6, "y", "y")
    
    format_axes([ax5, ax6])
    
selected_model_index = best_model
def button_pressed(event):
    fire_image, water_image = get_random_fire_water()
    global resized_fires, resized_waters
    resized_fires = [fire_image.resize((8,8)), fire_image.resize((16,16)),
                 fire_image.resize((32,32)), fire_image.resize((64,64)),
                 fire_image.resize((128,128))]
    resized_waters = [water_image.resize((8,8)), water_image.resize((16,16)),
                     water_image.resize((32,32)), water_image.resize((64,64)),
                     water_image.resize((128,128))]
    draw_ax1_ax2(fire_image, water_image)
    draw_ax5_ax6(selected_model_index)


# In[122]:


fig, ax1, ax2, ax3, ax4, ax_button, ax5, ax6 = create_grid()

main_title = '''How machine learning model distinguishes between Chinese handwritten Fire and Water
in different sizes
'''
set_main_title(main_title)

draw_ax1_ax2(fire_image, water_image)

draw_ax3(best_model)
ax3_original_facecolor = ax3.get_facecolor()

draw_ax4(best_model)
pos_ax4 = ax4.get_position() 
ax4.set_position([pos_ax4.x0, pos_ax4.y0 - 0.05,  pos_ax4.width, pos_ax4.height] )

myButton = Button(ax_button, 'Use other Fire/Water images', color='#34e5eb', hovercolor='#348feb')
myButton.label.set_fontsize('x-small')
myButton.on_clicked(button_pressed)

draw_ax5_ax6(best_model)

def onpick(event):
    global selected_model_index
    ind = event.ind
    model_index = ind[0]
    draw_ax3(model_index)
    draw_ax4(model_index)
    draw_ax5_ax6(model_index)
    selected_model_index = model_index
fig.canvas.mpl_connect('pick_event', onpick)
plt.show(block=True)
plt.ion()


# In[ ]:




