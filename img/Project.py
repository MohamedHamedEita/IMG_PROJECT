import tkinter
from tkinter import filedialog
from tkinter import *
from tkinter import ttk
import numpy as np
import PIL
from PIL import ImageTk
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from PIL.ImageFilter import SHARPEN, SMOOTH
import matplotlib.pyplot as plt
import cv2

window = Tk()   #make my window or root
#style = ttk.Style()   #set my style as classic
#style.theme_use("classic")
window.title("Image_Processing")   #set window title
window.configure(bg="#C67751") 
#window.iconphoto(False, PhotoImage(file='matlab_logo.png'))   #set window logo
window.geometry('750x780')   #set window size

#PROJECT GUI
# 1)PROJECT FRAMES according that our project consists of 1 rows and 4 columns
########################################################################################################################
f2 = LabelFrame(window, height=250, width=250, text="Original image", fg="#141515", background="#ABBAC1").grid(row=0, column=3, sticky='n', pady=10)
f3 = LabelFrame(window, height=100, width=140, text="Load image", fg="#141515",background="#C99AF4").grid(row=0, column=0, sticky='nw', padx=10, pady=30)
f4 = LabelFrame(window, height=100, width=140, text="Convert", fg="#141515").grid(row=0, column=1, sticky='nw', padx=5, pady=30)
f6 = LabelFrame(window, height=130, width=170, text="Point Transform Op's", fg="#141515", background="#CCD5D7").grid(row=0, column=0, columnspan=3,pady=135, padx=10, sticky='new')
f8 = LabelFrame(window, height=250, width=250, text="Result", fg="green", background="white").grid(row=0, column=3, sticky='n', pady=520)
f9 = LabelFrame(window, height=245, width=150, text="Local Transform Op's", fg="#141515",background="#BACCCC").grid(row=0, column=0, columnspan=3, sticky='new', padx=10, pady=270)
f10 = LabelFrame(window, height=245, width=150, text="Edge Detection", fg="#141515",background="#BACCCC").grid(row=0, column=1, columnspan=2, sticky='new', padx=20, pady=270)
f11 = LabelFrame(window, height=245, width=150, text="Global", fg="#141515",background="#BACCCC").grid(row=0, column=2, columnspan=2, sticky='new', padx=30, pady=270)



pil = 0
cv = 0
# 2)PROJECT BUTTONS
########################################################################################################################
def open_event():   #this function will be implemented when open button is clicked and return image path
    global filename
    filename = filedialog.askopenfilename(initialdir="/", title="Select a File", filetypes=(('Jpg Files', '.jpg'), ('PNG Files', '.png'),("Text files", "*.txt*"), ("all files", "*.*")))
    if len(filename) > 0:
        global image
        image = Image.open(filename)
        image = image.resize((235, 235))  # to resize image
        test = ImageTk.PhotoImage(image)
        label1 = tkinter.Label(image=test)
        label1.image = test
        label1.place(x=495, y=27)    #Position image
        DC_check.set(1)
    return 0
#Create button to browse in my computer to choose image
B_open = Button(f3, text="Open", width=15,fg="#141515", background="#DFCFF0", command=lambda: open_event()).grid(row=0, column=0, padx=20, pady=70, sticky='wn')


def Brightness_event():
    global pil
    pil = 1
    global cv
    cv = 0
    global img
    img = ImageEnhance.Brightness(image)
    img = img.enhance(1.5)
    img = img.resize((235, 235))  # to resize image
    test = ImageTk.PhotoImage(img)
    label1 = tkinter.Label(image=test)
    label1.image = test
    label1.place(x=495, y=535)  # Position image
    return 0
B_brightness = Button(f6, text="Brightness adjustment",  fg="white", background="#00303B",width=17, command =lambda: Brightness_event()).grid(row=0, column=0, columnspan=3, padx=25, pady=153, sticky='wn')


def Contrast_event():
    global pil
    pil = 1
    global cv
    cv = 0
    global img
    img = ImageEnhance.Contrast(image)
    img = img.enhance(1.5)
    test = ImageTk.PhotoImage(img)
    label1 = tkinter.Label(image=test)
    label1.image = test
    label1.place(x=495, y=535)  # Position image
    return 0
B_contrast = Button(f6, text="Contrast adjustment", width=17,fg="white", background="#00303B", command =lambda: Contrast_event()).grid(row=0, column=0, columnspan=3, padx=100, pady=181, sticky='wn')

def Histogram_event():
    img = cv2.imread(filename)
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()
    return 0
B_histogram = Button(f6, text="Histogram", width=17,fg="white", background="#00303B", command =lambda: Histogram_event()).grid(row=0, column=0, columnspan=3, padx=180, pady=208, sticky='wn')

def Histogram_Equalization_event():
    global pil
    pil = 1
    global cv
    cv = 0
    global img
    img = ImageOps.equalize(image, mask=None)
    test = ImageTk.PhotoImage(img)
    label1 = tkinter.Label(image=test)
    label1.image = test
    label1.place(x=495, y=535)
    return 0
B_histogram_equalization = Button(f6, text="Histogram Equalization", width=17,fg="white", background="#00303B", command =lambda: Histogram_Equalization_event()).grid(row=0, column=1,columnspan=2, padx=100, pady=235, sticky='wn')

def Low_pass_event():
    global pil
    pil = 1
    global cv
    cv = 0
    global img
    img = image.filter(ImageFilter.BLUR)
    test = ImageTk.PhotoImage(img)
    label1 = tkinter.Label(image=test)
    label1.image = test
    label1.place(x=495, y=535)
    return 0
B_low_pass = Button(f9, text="Low pass filter", width=17, fg="white", background="#1A5758",command =lambda: Low_pass_event()).grid(row=0, column=0,columnspan=2, padx=20, pady=300, sticky='wn')

def High_pass_event():
    global pil
    pil = 1
    global cv
    cv = 0
    global img
    img = image.filter(SHARPEN)
    test = ImageTk.PhotoImage(img)
    label1 = tkinter.Label(image=test)
    label1.image = test
    label1.place(x=495, y=535)
    return 0
B_high_pass = Button(f9, text="High pass filter", width=17,fg="white", background="#1A5758", command =lambda: High_pass_event()).grid(row=0, column=0,columnspan=2, padx=20, pady=350, sticky='wn')

def Median_filter_event():
    global pil
    pil = 1
    global cv
    cv = 0
    global img
    img = image.filter(ImageFilter.MedianFilter)
    test = ImageTk.PhotoImage(img)
    label1 = tkinter.Label(image=test)
    label1.image = test
    label1.place(x=495, y=535)
    return 0
B_median_filter = Button(f9, text="Median filtering", width=17,fg="white", background="#1A5758", command =lambda: Median_filter_event()).grid(row=0, column=0,columnspan=2, padx=20, pady=400, sticky='wn')#1A5758

def Averaging_filter_event():
    global pil
    pil = 1
    global cv
    cv = 0
    global img
    img = image.filter(SMOOTH)
    img = img.resize((235, 235))
    test = ImageTk.PhotoImage(img)
    label1 = tkinter.Label(image=test)
    label1.image = test
    label1.place(x=495, y=535)
    return 0
B_averaging_filter = Button(f9, text="Averaging filtering", width=17,fg="white", background="#1A5758", command=lambda: Averaging_filter_event()).grid(row=0, column=0, columnspan=2, padx=20, pady=450, sticky='wn')

def apply_prewitt_edge_detector():
    global pil
    pil = 1
    global cv
    cv = 0
    global img
    gray_image = image.convert("L")  # Convert the image to grayscale
    img_array = np.array(gray_image)  # Convert to a NumPy array
    
    prewitt_x = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=3)
    prewitt_y = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=3)
    
    prewitt_image = np.sqrt(prewitt_x*2 + prewitt_y*2)
    
    prewitt_image = cv2.normalize(prewitt_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Update the image display with the Prewitt edge-detected image
    test = ImageTk.PhotoImage(image=Image.fromarray(prewitt_image))
    label1 = tkinter.Label(image=test)
    label1.image = test
    label1.configure(image=test)
    label1.image = test
    label1.place(x=495, y=535)
    return 0
B_averaging_filter = Button(f10, text="Prewitt_edge_detector", width=17,fg="white", background="#1A5758", command=lambda: apply_prewitt_edge_detector()).grid(row=0, column=1, columnspan=2, padx=30, pady=300, sticky='wn')

def apply_sobel_edge_detector(image):
    global pil
    pil = 1
    global cv
    cv = 0
    global img
    # Convert the image to grayscale
    gray_image = image.convert("L")
    
    # Convert to a NumPy array
    img_array = np.array(gray_image)
    
    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine the gradients
    sobel_image = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize the image
    sobel_image = cv2.normalize(sobel_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Convert NumPy array back to PIL Image
    sobel_image_pil = Image.fromarray(sobel_image)
    
    # Display the image (assuming Tkinter GUI)
    test = ImageTk.PhotoImage(image=sobel_image_pil)
    label1 = tkinter.Label(image=test)
    label1.image = test
    label1.configure(image=test)
    label1.image = test
    label1.place(x=495, y=535)
    
    # Return 0 as a placeholder
    return 0
B_averaging_filter = Button(f10, text="sobel_edge_detector", width=17,fg="white", background="#1A5758", command=lambda: apply_sobel_edge_detector()).grid(row=0, column=1, columnspan=2, padx=30, pady=350, sticky='wn')
def Erosion_event():
    global pil
    pil = 0
    global cv
    cv = 1
    global img
    img = cv2.imread(filename, 0)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    #cv2.imshow("Eroded", erosion)

    width = 235
    height = 235
    dim = (width, height)
    # resize image
    resized = cv2.resize(erosion, dim, interpolation=cv2.INTER_AREA)
    img = resized
    height, width = resized.shape
    canvas = tkinter.Canvas(window, width=width, height=height)
    canvas.place(x=495, y=535)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(resized))
    canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
    window.mainloop()

    return 0
B_erosion = Button(f10, text="Erosion", width=17,fg="white", background="#1A5758", command =lambda: Erosion_event()).grid(row=0, column=2,columnspan=3, padx=35, pady=300, sticky='wn')
def Dilation_event():
    global pil
    pil = 0
    global cv
    cv = 1
    global img
    img = cv2.imread(filename, 0)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)

    width = 235
    height = 235
    dim = (width, height)
    # resize image
    resized = cv2.resize(dilation, dim, interpolation=cv2.INTER_AREA)
    img = resized
    height, width = resized.shape
    canvas = tkinter.Canvas(window, width=width, height=height)
    canvas.place(x=495, y=535)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(resized))
    canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
    window.mainloop()
    return 0
B_dilation = Button(f10, text="Dilation", width=17,fg="white", background="#1A5758", command =lambda: Dilation_event()).grid(row=0, column=2,columnspan=3, padx=35, pady=450, sticky='wn')
def Circles_detection_event():
    global pil
    pil = 0
    global cv
    cv = 1
    global img
    img = cv2.imread(filename, 0)
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    img = cimg
    #cv2.imshow('detected circles', cimg)
    width = 235
    height = 235
    dim = (width, height)
    # resize image
    resized = cv2.resize(cimg, dim, interpolation=cv2.INTER_AREA)

    height, width, no_channels = resized.shape
    canvas = tkinter.Canvas(window, width=width, height=height)
    canvas.place(x=495, y=535)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(resized))
    canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
    window.mainloop()

    return 0
B_circles_detection = Button(f11, text="Circles detection using HT",width=19,fg="white", background="#1A5758", command =lambda: Circles_detection_event()).grid(row=0, column=1,columnspan=2, padx=30, pady=400, sticky='wn')
def Close_event():
    global pil
    pil = 0
    global cv
    cv = 1
    global img
    img = cv2.imread(filename, 0)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    width = 235
    height = 235
    dim = (width, height)
    # resize image
    resized = cv2.resize(closing, dim, interpolation=cv2.INTER_AREA)
    img = resized
    height, width = resized.shape
    canvas = tkinter.Canvas(window, width=width, height=height)
    canvas.place(x=495, y=535)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(resized))
    canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
    window.mainloop()

    return 0
B_close = Button(f11, text="Close", width=8, fg="white", background="#1A5758",command =lambda: Close_event()).grid(row=0, column=2, columnspan=3, padx=35, pady=350, sticky='wn')
def Open2_event():
    global pil
    pil = 0
    global cv
    cv = 1
    global img
    img = cv2.imread(filename, 0)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    width = 235
    height = 235
    dim = (width, height)
    # resize image
    resized = cv2.resize(opening, dim, interpolation=cv2.INTER_AREA)
    img = resized
    height, width = resized.shape
    canvas = tkinter.Canvas(window, width=width, height=height)
    canvas.place(x=495, y=535)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(resized))
    canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
    window.mainloop()
    return 0
B_open2 = Button(f11, text="Open", width=8, fg="white", background="#1A5758",command=lambda: Open2_event()).grid(row=0, column=2, columnspan=3, padx=35, pady=400, sticky='wn')


def Save_event():
    filename2 = filedialog.asksaveasfilename(initialdir="/", defaultextension=".png", title="Select file", filetypes=(("PNG file", "*.png"), ("All Files", "*.*")))
    global img
    print(filename2)
    if (cv==1):
        cv2.imwrite(filename2, img)
    elif (pil==1):
        img.save(filename2)
B_save = Button(window, text="Save Result image", width=15, command=lambda: Save_event()).grid(row=0, column=0, columnspan=3, padx=70, pady=730, sticky='wn')

def Exit_event():
    exit()
    return 0
B_exit = Button(window, text="Exit", width=15,fg="white",background="#A10612", command=lambda: Exit_event()).grid(row=0, column=2, pady=730, sticky='wn')


# 3)PROJECT CHECK BUTTONS
########################################################################################################################
DC_check = IntVar()   #variable of default color check
DC_check.set(0)       #set the initial value of check button 0
def DC_event():
    if DC_check.get() == 1:
        Gray_check.set(0)
        rgbimg = image
        rgbimg = rgbimg.resize((235, 235))
        test = ImageTk.PhotoImage(rgbimg)
        label1 = tkinter.Label(image=test)
        label1.image = test
        label1.place(x=495, y=27)
    if DC_check.get() == 0:
        Gray_check.set(1)
        Gray_event()
#Create check button to choose the default color
DC_checkbutton = Checkbutton(f4, text="Default Color", variable=DC_check, onvalue=1, offvalue=0, command=lambda: DC_event()).grid(row=0, column=1, padx=20, pady=50, sticky='wn')

Gray_check = IntVar()
Gray_check.set(0)
def Gray_event():
    if Gray_check.get() == 1:
        DC_check.set(0)
        global gray
        gray = image.convert('L')
        gray = gray.resize((235, 235))  # to resize image
        test = ImageTk.PhotoImage(gray)
        label1 = tkinter.Label(image=test)
        label1.image = test
        # Position image
        label1.place(x=495, y=27)
    elif Gray_check.get() == 0:
        DC_check.set(1)
        DC_event()
#Create check button to choose the gray color
Gray_checkbutton = Checkbutton(f4, text="Gray Color", variable=Gray_check, onvalue=1, offvalue=0, command=lambda: Gray_event()).grid(row=0, column=1, padx=20, pady=90, sticky='wn')








window.mainloop()
