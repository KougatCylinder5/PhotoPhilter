# Cory Mavis
import cv2
import numpy
import os.path
import os
import ctypes
import subprocess

import tkinter
from tkinter import filedialog 

dir = os.path.join("C:\\","User",os.path.expanduser("~"),"AppData","Roaming","PhotoPhilter")

if not os.path.exists(dir):
    os.mkdir(dir)
if not os.path.exists(os.path.join(dir,"instructions.txt")):
    instructions = open(os.path.join(dir,"instructions.txt"),"a+")
    instructions.write("1.) Select Image File To Open \n2.) Select Preset File To Open (These are located at %appdata%\\Roaming\\PhotoPhiler\\Saves) \n3.) Adjust Sliders Until Desired Effect Has Been Achieved\n   a.) Each Settings Window contains the layers that they contain in the name of the window\n   b.) C# Red changes the red value, C# Green changes the green value, C# Blue changes the blue value, C# GS HR changes the highest value of grey changed,\n   and C# GS LS changes the lowest value of grey changed\n4.) Press Either W or S, W will save the current sliders to a preset. S will open a prompt to save the image to a desired location\n5.) Press E or Esc to close the program")
    print(instructions)
    instructions.close()
process = subprocess.Popen(["notepad.exe", os.path.join(dir,"instructions.txt")]) 

blank = numpy.zeros((6,5,1),numpy.uint8)

if not os.path.exists(os.path.join(dir,"saves")):
    os.mkdir(os.path.join(dir,"saves"))
if not os.path.exists(os.path.join(dir,"output")):
    os.mkdir(os.path.join(dir,"output"))
if not os.path.exists(os.path.join(dir,"saves","Blank.png")):
    cv2.imwrite(os.path.join(dir,"saves","Blank.png"),blank)

tkinter.Tk().withdraw() # prevents an empty tkinter window from appearing
filename = filedialog.askopenfile(filetypes = [(".png, .jpg, .jpeg, .jfif",".png; .jpg; .jpeg; .jfif")],title = "Select Image File To Open",initialdir = os.path.join("C:\\","User",os.path.expanduser("~"),"Downloads"),multiple = False).name

user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

tkinter.Tk().withdraw()
saveName = filedialog.askopenfile(filetypes = [(".png",".png")],title = "Select Preset File To Open",initialdir = os.path.join(dir,"saves"),multiple = False)
print(saveName)

if(saveName == None or saveName.name == "Blank.png"):
    fileTrue = False
else:
    fileTrue = True
    saveName = saveName.name


original_image = cv2.imread(filename,1)
original_image_greyscale = cv2.imread(filename,0)
original_image_greyscale = cv2.cvtColor(original_image_greyscale, cv2.COLOR_GRAY2BGR)
save_file = ""
save_file = cv2.imread(saveName,0)

rSH = ((user32.GetSystemMetrics(0) - 20)/original_image.shape[1] * 100)/3
rSV = ((user32.GetSystemMetrics(1) - 20)/original_image.shape[0] * 100)/3
if(min(rSH,rSV) > 100):
    rS = 100
else:
    rS = min(rSH,rSV)

scale_percent =  rS# percent of original size
width = int(original_image.shape[1] * scale_percent / 100)
height = int(original_image.shape[0] * scale_percent / 100)
dim = (width, height)
original_image = cv2.resize(original_image, dim, interpolation = cv2.INTER_AREA)
width_grey = int(original_image_greyscale.shape[1] * scale_percent / 100)
height_grey = int(original_image_greyscale.shape[0] * scale_percent / 100)
dim_grey = (width_grey, height_grey)
original_image_greyscale = cv2.resize(original_image_greyscale, dim_grey, interpolation = cv2.INTER_AREA)
#scales down too large images so that they fit on screen

cv2.namedWindow("Settings Window (Colors 1 - 3)")
cv2.namedWindow("Settings Window (Colors 4 - 6)")
cv2.namedWindow("Image Overlay")

image_height = original_image.shape[0]
image_width = original_image.shape[1]
image_channels = original_image.shape[2]

original = numpy.zeros((image_height * 3 + 20,image_width * 3 + 20,image_channels), numpy.uint8)
grey = numpy.zeros((image_height * 3 + 20,image_width * 3 + 20,image_channels), numpy.uint8)
y = 0
x = 0

while x < image_width :
    while y < image_height:
        original[y][x] = original_image[y][x]
        y = y + 1
    x = int(x) + 1
    y = 0
y = 0
x = image_width + 10
while x < image_width * 2 + 10:
    while y < image_height:
        grey[y][x] = original_image_greyscale[y][x - (image_width + 10)]
        y = y + 1
    x = int(x) + 1
    y = 0

setting = numpy.zeros((50,950,image_channels), numpy.uint8)
setting[0:50,0:950,0:image_channels] = [210,210,210]
cv2.imshow("Settings Window (Colors 1 - 3)",setting)
cv2.imshow("Settings Window (Colors 4 - 6)",setting)






def reload():
    
    global output
    
    colorOne = numpy.zeros((image_height,image_width,image_channels), numpy.uint8)
    colorTwo = numpy.zeros((image_height,image_width,image_channels), numpy.uint8)
    colorThree = numpy.zeros((image_height,image_width,image_channels), numpy.uint8)
    colorFour = numpy.zeros((image_height,image_width,image_channels), numpy.uint8)
    colorFive = numpy.zeros((image_height,image_width,image_channels), numpy.uint8)
    colorSix = numpy.zeros((image_height,image_width,image_channels), numpy.uint8)
    #creates mask arrays
    
    colorOne[0:image_height,0:image_width, 0:image_channels] = [CB1,CG1,CR1]
    colorTwo[0:image_height,0:image_width, 0:image_channels] = [CB2,CG2,CR2]
    colorThree[0:image_height,0:image_width, 0:image_channels] = [CB3,CG3,CR3]
    colorFour[0:image_height,0:image_width, 0:image_channels] = [CB4,CG4,CR4]
    colorFive[0:image_height,0:image_width, 0:image_channels] = [CB5,CG5,CR5]
    colorSix[0:image_height,0:image_width, 0:image_channels] = [CB6,CG6,CR6]
    #assigns colors to the full mask
    
    min_grey_for_1 = [CGS1LR,CGS1LR,CGS1LR]
    max_grey_for_1 = [CGS1HR,CGS1HR,CGS1HR]
    min_grey_for_2 = [CGS2LR,CGS2LR,CGS2LR]
    max_grey_for_2 = [CGS2HR,CGS2HR,CGS2HR]
    min_grey_for_3 = [CGS3LR,CGS3LR,CGS3LR]
    max_grey_for_3 = [CGS3HR,CGS3HR,CGS3HR]
    min_grey_for_4 = [CGS4LR,CGS4LR,CGS4LR]
    max_grey_for_4 = [CGS4HR,CGS4HR,CGS4HR]
    min_grey_for_5 = [CGS5LR,CGS5LR,CGS5LR]
    max_grey_for_5 = [CGS5HR,CGS5HR,CGS5HR]
    min_grey_for_6 = [CGS6LR,CGS6LR,CGS6LR]
    max_grey_for_6 = [CGS6HR,CGS6HR,CGS6HR]
    #creates range for max
    
    min_grey_for_1 = numpy.array(min_grey_for_1,dtype = "uint8")
    max_grey_for_1 = numpy.array(max_grey_for_1,dtype = "uint8")
    min_grey_for_2 = numpy.array(min_grey_for_2,dtype = "uint8")
    max_grey_for_2 = numpy.array(max_grey_for_2,dtype = "uint8")
    min_grey_for_3 = numpy.array(min_grey_for_3,dtype = "uint8")
    max_grey_for_3 = numpy.array(max_grey_for_3,dtype = "uint8")
    min_grey_for_4 = numpy.array(min_grey_for_4,dtype = "uint8")
    max_grey_for_4 = numpy.array(max_grey_for_4,dtype = "uint8")
    min_grey_for_5 = numpy.array(min_grey_for_5,dtype = "uint8")
    max_grey_for_5 = numpy.array(max_grey_for_5,dtype = "uint8")
    min_grey_for_6 = numpy.array(min_grey_for_6,dtype = "uint8")
    max_grey_for_6 = numpy.array(max_grey_for_6,dtype = "uint8")
    #fixes the data type for ranges
    
    block_all_but_C1 = cv2.inRange(original_image_greyscale, min_grey_for_1, max_grey_for_1)
    block_all_but_C2 = cv2.inRange(original_image_greyscale, min_grey_for_2, max_grey_for_2)
    block_all_but_C3 = cv2.inRange(original_image_greyscale, min_grey_for_3, max_grey_for_3)
    block_all_but_C4 = cv2.inRange(original_image_greyscale, min_grey_for_4, max_grey_for_4)
    block_all_but_C5 = cv2.inRange(original_image_greyscale, min_grey_for_5, max_grey_for_5)
    block_all_but_C6 = cv2.inRange(original_image_greyscale, min_grey_for_6, max_grey_for_6)
    #creates mask
    
    colorOneParts = cv2.bitwise_or(colorOne, colorOne, mask = block_all_but_C1)
    colorTwoParts = cv2.bitwise_or(colorTwo, colorTwo, mask = block_all_but_C2)
    colorThreeParts = cv2.bitwise_or(colorThree, colorThree, mask = block_all_but_C3)
    colorFourParts = cv2.bitwise_or(colorFour, colorFour, mask = block_all_but_C4)
    colorFiveParts = cv2.bitwise_or(colorFive, colorFive, mask = block_all_but_C5)
    colorSixParts = cv2.bitwise_or(colorSix, colorSix, mask = block_all_but_C6)
    #combines the full mask and the mask to create a partal image
    
    final = numpy.zeros((image_height * 3 + 20,image_width * 3 + 20, image_channels), numpy.uint8)
    final[0:image_height * 3 + 20,image_width:image_width + 10,0:image_channels] = [255,255,255]
    final[0:image_height * 3 + 20,image_width * 2 + 10:image_width * 2 + 20,0:image_channels] = [255,255,255]
    final[image_height:image_height + 10,0:image_width * 3 + 20,0:image_channels] = [255,255,255]
    final[image_height * 2 + 10:image_height * 3 + 20,0:image_width * 3 + 20,0:image_channels] = [255,255,255]
    #creates the final image array
    
    final_image = numpy.zeros((image_height * 3 + 20,image_width * 3 + 20, image_channels), numpy.uint8)
    #creates the custom image array that the color#Parts will be overlayed on
    
    final[image_height + 10:image_height * 2 + 10,0:image_width] = colorOneParts
    final[image_height + 10:image_height * 2 + 10,image_width + 10:image_width * 2 + 10] = colorTwoParts
    final[image_height + 10:image_height * 2 + 10,image_width * 2 + 20:image_width * 3 + 20] = colorThreeParts
    final[image_height * 2 + 20:image_height * 3 + 20,0:image_width] = colorFourParts
    final[image_height * 2 + 20:image_height * 3 + 20,image_width + 10:image_width * 2 + 10] = colorFiveParts
    final[image_height * 2 + 20:image_height * 3 + 20,image_width * 2 + 20:image_width * 3 + 20] = colorSixParts
    #merge the masks together onto the proper locations in the array
    
    output = cv2.bitwise_or(colorOneParts,colorTwoParts)
    output = cv2.bitwise_or(colorThreeParts,output)
    output = cv2.bitwise_or(colorFourParts,output)
    output = cv2.bitwise_or(colorFiveParts,output)
    output = cv2.bitwise_or(colorSixParts,output)
    
    combined_image = numpy.zeros((image_height * 3 + 20,image_width * 3 + 20, image_channels), numpy.uint8)
    
    final_image[0:image_height,image_width * 2 + 20:image_width * 3 + 20] = colorOneParts
    combined_image = cv2.bitwise_or(final_image,combined_image)
    final_image[0:image_height,image_width * 2 + 20:image_width * 3 + 20] = colorTwoParts
    combined_image = cv2.bitwise_or(final_image,combined_image)
    final_image[0:image_height,image_width * 2 + 20:image_width * 3 + 20] = colorThreeParts
    combined_image = cv2.bitwise_or(final_image,combined_image)
    final_image[0:image_height,image_width * 2 + 20:image_width * 3 + 20] = colorFourParts
    combined_image = cv2.bitwise_or(final_image,combined_image)
    final_image[0:image_height,image_width * 2 + 20:image_width * 3 + 20] = colorFiveParts
    combined_image = cv2.bitwise_or(final_image,combined_image)
    final_image[0:image_height,image_width * 2 + 20:image_width * 3 + 20] = colorSixParts
    combined_image = cv2.bitwise_or(final_image,combined_image)
    #merge the Parts of images onto a single array in the top right hand corner
    
    final = cv2.addWeighted(combined_image, 1, final, 1,1)
    final = cv2.addWeighted(final,1,grey,1,1)
    final = cv2.addWeighted(final,1,original,1,1)
    cv2.imshow("Image Overlay",final)



def saveImage():
    saveImage = numpy.zeros((6,5), numpy.uint8)
    #cv2.getTrackbarPos( ,"Settings Window")
    saveImage[0,0] = cv2.getTrackbarPos("C1 Red","Settings Window (Colors 1 - 3)")
    saveImage[0,1] = cv2.getTrackbarPos("C1 Green","Settings Window (Colors 1 - 3)")
    saveImage[0,2] = cv2.getTrackbarPos("C1 Blue" ,"Settings Window (Colors 1 - 3)")
    saveImage[0,3] = cv2.getTrackbarPos("C1 GS HR" ,"Settings Window (Colors 1 - 3)")
    saveImage[0,4] = cv2.getTrackbarPos("C1 GS LR" ,"Settings Window (Colors 1 - 3)")
    saveImage[1,0] = cv2.getTrackbarPos("C2 Red","Settings Window (Colors 1 - 3)")
    saveImage[1,1] = cv2.getTrackbarPos("C2 Green","Settings Window (Colors 1 - 3)")
    saveImage[1,2] = cv2.getTrackbarPos("C2 Blue" ,"Settings Window (Colors 1 - 3)")
    saveImage[1,3] = cv2.getTrackbarPos("C2 GS HR" ,"Settings Window (Colors 1 - 3)")
    saveImage[1,4] = cv2.getTrackbarPos("C2 GS LR" ,"Settings Window (Colors 1 - 3)")
    saveImage[2,0] = cv2.getTrackbarPos("C3 Red","Settings Window (Colors 1 - 3)")
    saveImage[2,1] = cv2.getTrackbarPos("C3 Green","Settings Window (Colors 1 - 3)")
    saveImage[2,2] = cv2.getTrackbarPos("C3 Blue" ,"Settings Window (Colors 1 - 3)")
    saveImage[2,3] = cv2.getTrackbarPos("C3 GS HR" ,"Settings Window (Colors 1 - 3)")
    saveImage[2,4] = cv2.getTrackbarPos("C3 GS LR" ,"Settings Window (Colors 1 - 3)")
    saveImage[3,0] = cv2.getTrackbarPos("C4 Red","Settings Window (Colors 4 - 6)")
    saveImage[3,1] = cv2.getTrackbarPos("C4 Green","Settings Window (Colors 4 - 6)")
    saveImage[3,2] = cv2.getTrackbarPos("C4 Blue" ,"Settings Window (Colors 4 - 6)")
    saveImage[3,3] = cv2.getTrackbarPos("C4 GS HR" ,"Settings Window (Colors 4 - 6)")
    saveImage[3,4] = cv2.getTrackbarPos("C4 GS LR" ,"Settings Window (Colors 4 - 6)")
    saveImage[4,0] = cv2.getTrackbarPos("C5 Red","Settings Window (Colors 4 - 6)")
    saveImage[4,1] = cv2.getTrackbarPos("C5 Green","Settings Window (Colors 4 - 6)")
    saveImage[4,2] = cv2.getTrackbarPos("C5 Blue" ,"Settings Window (Colors 4 - 6)")
    saveImage[4,3] = cv2.getTrackbarPos("C5 GS HR" ,"Settings Window (Colors 4 - 6)")
    saveImage[4,4] = cv2.getTrackbarPos("C5 GS LR" ,"Settings Window (Colors 4 - 6)")
    saveImage[5,0] = cv2.getTrackbarPos("C6 Red","Settings Window (Colors 4 - 6)")
    saveImage[5,1] = cv2.getTrackbarPos("C6 Green","Settings Window (Colors 4 - 6)")
    saveImage[5,2] = cv2.getTrackbarPos("C6 Blue" ,"Settings Window (Colors 4 - 6)")
    saveImage[5,3] = cv2.getTrackbarPos("C6 GS HR" ,"Settings Window (Colors 4 - 6)")
    saveImage[5,4] = cv2.getTrackbarPos("C6 GS LR" ,"Settings Window (Colors 4 - 6)")
    return saveImage



def Color1R(v):
    global CR1
    CR1 = v
    reload()
def Color1G(v):
    global CG1
    CG1 = v
    reload()
def Color1B(v):
    global CB1
    CB1 = v
    reload()
def Color1GSLR(v):
    global CGS1LR
    CGS1LR = v
    reload()
def Color1GSHR(v):
    global CGS1HR
    CGS1HR = v
    reload()
def Color2R(v):
    global CR2
    CR2 = v
    reload()
def Color2G(v):
    global CG2
    CG2 = v
    reload()
def Color2B(v):
    global CB2
    CB2 = v
    reload()
def Color2GSLR(v):
    global CGS2LR
    CGS2LR = v
    reload()
def Color2GSHR(v):
    global CGS2HR
    CGS2HR = v
    reload()
def Color3R(v):
    global CR3
    CR3 = v
    reload()
def Color3G(v):
    global CG3
    CG3 = v
    reload()
def Color3B(v):
    global CB3
    CB3 = v
    reload()
def Color3GSLR(v):
    global CGS3LR
    CGS3LR = v
    reload()
def Color3GSHR(v):
    global CGS3HR
    CGS3HR = v
    reload()
def Color4R(v):
    global CR4
    CR4 = v
    reload()
def Color4G(v):
    global CG4
    CG4 = v
    reload()
def Color4B(v):
    global CB4
    CB4 = v
    reload()
def Color4GSLR(v):
    global CGS4LR
    CGS4LR = v
    reload()
def Color4GSHR(v):
    global CGS4HR
    CGS4HR = v
    reload()
def Color5R(v):
    global CR5
    CR5 = v
    reload()
def Color5G(v):
    global CG5
    CG5 = v
    reload()
def Color5B(v):
    global CB5
    CB5 = v
    reload()
def Color5GSLR(v):
    global CGS5LR
    CGS5LR = v
    reload()
def Color5GSHR(v):
    global CGS5HR
    CGS5HR = v
    reload()
def Color6R(v):
    global CR6
    CR6 = v
    reload()
def Color6G(v):
    global CG6
    CG6 = v
    reload()
def Color6B(v):
    global CB6
    CB6 = v
    reload()
def Color6GSLR(v):
    global CGS6LR
    CGS6LR = v
    reload()
def Color6GSHR(v):
    global CGS6HR
    CGS6HR = v
    reload()

cv2.createTrackbar("C1 Red","Settings Window (Colors 1 - 3)",0,255,Color1R)
cv2.createTrackbar("C1 Green","Settings Window (Colors 1 - 3)",0,255,Color1G)
cv2.createTrackbar("C1 Blue","Settings Window (Colors 1 - 3)",0,255,Color1B)
cv2.createTrackbar("C1 GS HR","Settings Window (Colors 1 - 3)",0,255,Color1GSHR)
cv2.createTrackbar("C1 GS LR","Settings Window (Colors 1 - 3)",0,255,Color1GSLR)
cv2.createTrackbar("C2 Red","Settings Window (Colors 1 - 3)",0,255,Color2R)
cv2.createTrackbar("C2 Green","Settings Window (Colors 1 - 3)",0,255,Color2G)
cv2.createTrackbar("C2 Blue","Settings Window (Colors 1 - 3)",0,255,Color2B)
cv2.createTrackbar("C2 GS HR","Settings Window (Colors 1 - 3)",0,255,Color2GSHR)
cv2.createTrackbar("C2 GS LR","Settings Window (Colors 1 - 3)",0,255,Color2GSLR)
cv2.createTrackbar("C3 Red","Settings Window (Colors 1 - 3)",0,255,Color3R)
cv2.createTrackbar("C3 Green","Settings Window (Colors 1 - 3)",0,255,Color3G)
cv2.createTrackbar("C3 Blue","Settings Window (Colors 1 - 3)",0,255,Color3B)
cv2.createTrackbar("C3 GS HR","Settings Window (Colors 1 - 3)",0,255,Color3GSHR)
cv2.createTrackbar("C3 GS LR","Settings Window (Colors 1 - 3)",0,255,Color3GSLR)
cv2.createTrackbar("C4 Red","Settings Window (Colors 4 - 6)",0,255,Color4R)
cv2.createTrackbar("C4 Green","Settings Window (Colors 4 - 6)",0,255,Color4G)
cv2.createTrackbar("C4 Blue","Settings Window (Colors 4 - 6)",0,255,Color4B)
cv2.createTrackbar("C4 GS HR","Settings Window (Colors 4 - 6)",0,255,Color4GSHR)
cv2.createTrackbar("C4 GS LR","Settings Window (Colors 4 - 6)",0,255,Color4GSLR)
cv2.createTrackbar("C5 Red","Settings Window (Colors 4 - 6)",0,255,Color5R)
cv2.createTrackbar("C5 Green","Settings Window (Colors 4 - 6)",0,255,Color5G)
cv2.createTrackbar("C5 Blue","Settings Window (Colors 4 - 6)",0,255,Color5B)
cv2.createTrackbar("C5 GS HR","Settings Window (Colors 4 - 6)",0,255,Color5GSHR)
cv2.createTrackbar("C5 GS LR","Settings Window (Colors 4 - 6)",0,255,Color5GSLR)
cv2.createTrackbar("C6 Red","Settings Window (Colors 4 - 6)",0,255,Color6R)
cv2.createTrackbar("C6 Green","Settings Window (Colors 4 - 6)",0,255,Color6G)
cv2.createTrackbar("C6 Blue","Settings Window (Colors 4 - 6)",0,255,Color6B)
cv2.createTrackbar("C6 GS HR","Settings Window (Colors 4 - 6)",0,255,Color6GSHR)
cv2.createTrackbar("C6 GS LR","Settings Window (Colors 4 - 6)",0,255,Color6GSLR)


if(fileTrue == False):
    CR1 = 0
    CG1 = 0
    CB1 = 0
    CGS1HR = 0
    CGS1LR = 0
    CR2 = 0
    CG2 = 0
    CB2 = 0
    CGS2HR = 0
    CGS2LR = 0
    CR3 = 0
    CG3 = 0
    CB3 = 0
    CGS3HR = 0
    CGS3LR = 0
    CR4 = 0
    CG4 = 0
    CB4 = 0
    CGS4HR = 0
    CGS4LR = 0
    CR5 = 0
    CG5 = 0
    CB5 = 0
    CGS5HR = 0
    CGS5LR = 0
    CR6 = 0
    CG6 = 0
    CB6 = 0
    CGS6HR = 0
    CGS6LR = 0
else:
    
    CR1 = save_file[0,0]
    cv2.setTrackbarPos("C1 Red" ,"Settings Window (Colors 1 - 3)",save_file[0,0])
    CG1 = save_file[0,1]
    cv2.setTrackbarPos("C1 Green" ,"Settings Window (Colors 1 - 3)",save_file[0,1])
    CB1 = save_file[0,2]
    cv2.setTrackbarPos("C1 Blue" ,"Settings Window (Colors 1 - 3)",save_file[0,2])
    CGS1HR = save_file[0,3]
    cv2.setTrackbarPos("C1 GS HR" ,"Settings Window (Colors 1 - 3)",save_file[0,3])
    CGS1LR = save_file[0,4]
    cv2.setTrackbarPos("C1 GS LR" ,"Settings Window (Colors 1 - 3)",save_file[0,4])
    CR2 = save_file[1,0]
    cv2.setTrackbarPos("C2 Red" ,"Settings Window (Colors 1 - 3)",save_file[1,0])
    CG2 = save_file[1,1]
    cv2.setTrackbarPos("C2 Green" ,"Settings Window (Colors 1 - 3)",save_file[1,1])
    CB2 = save_file[1,2]
    cv2.setTrackbarPos("C2 Blue" ,"Settings Window (Colors 1 - 3)",save_file[1,2])
    CGS2HR = save_file[1,3]
    cv2.setTrackbarPos("C2 GS HR" ,"Settings Window (Colors 1 - 3)",save_file[1,3])
    CGS2LR = save_file[1,4]
    cv2.setTrackbarPos("C2 GS LR" ,"Settings Window (Colors 1 - 3)",save_file[1,4])
    CR3 = save_file[2,0]
    cv2.setTrackbarPos("C3 Red" ,"Settings Window (Colors 1 - 3)",save_file[2,0])
    CG3 = save_file[2,1]
    cv2.setTrackbarPos("C3 Green" ,"Settings Window (Colors 1 - 3)",save_file[2,1])
    CB3 = save_file[2,2]
    cv2.setTrackbarPos("C3 Blue" ,"Settings Window (Colors 1 - 3)",save_file[2,2])
    CGS3HR = save_file[2,3]
    cv2.setTrackbarPos("C3 GS HR" ,"Settings Window (Colors 1 - 3)",save_file[2,3])
    CGS3LR = save_file[2,4]
    cv2.setTrackbarPos("C3 GS LR" ,"Settings Window (Colors 1 - 3)",save_file[2,4])
    CR4 = save_file[3,0]
    cv2.setTrackbarPos("C4 Red" ,"Settings Window (Colors 4 - 6)",save_file[3,0])
    CG4 = save_file[3,1]
    cv2.setTrackbarPos("C4 Green" ,"Settings Window (Colors 4 - 6)",save_file[3,1])
    CB4 = save_file[3,2]
    cv2.setTrackbarPos("C4 Blue" ,"Settings Window (Colors 4 - 6)",save_file[3,2])
    CGS4HR = save_file[3,3]
    cv2.setTrackbarPos("C4 GS HR" ,"Settings Window (Colors 4 - 6)",save_file[3,3])
    CGS4LR = save_file[3,4]
    cv2.setTrackbarPos("C4 GS LR" ,"Settings Window (Colors 4 - 6)",save_file[3,4])
    CR5 = save_file[4,0]
    cv2.setTrackbarPos("C5 Red" ,"Settings Window (Colors 4 - 6)",save_file[4,0])
    CG5 = save_file[4,1]
    cv2.setTrackbarPos("C5 Green" ,"Settings Window (Colors 4 - 6)",save_file[4,1])
    CB5 = save_file[4,2]
    cv2.setTrackbarPos("C5 Blue" ,"Settings Window (Colors 4 - 6)",save_file[4,2])
    CGS5HR = save_file[4,3]
    cv2.setTrackbarPos("C5 GS HR" ,"Settings Window (Colors 4 - 6)",save_file[4,3])
    CGS5LR = save_file[4,4]
    cv2.setTrackbarPos("C5 GS LR" ,"Settings Window (Colors 4 - 6)",save_file[4,4])
    CR6 = save_file[5,0]
    cv2.setTrackbarPos("C6 Red" ,"Settings Window (Colors 4 - 6)",save_file[5,0])
    CG6 = save_file[5,1]
    cv2.setTrackbarPos("C6 Green" ,"Settings Window (Colors 4 - 6)",save_file[5,1])
    CB6 = save_file[5,2]
    cv2.setTrackbarPos("C6 Blue" ,"Settings Window (Colors 4 - 6)",save_file[5,2])
    CGS6HR = save_file[5,3]
    cv2.setTrackbarPos("C6 GS HR" ,"Settings Window (Colors 4 - 6)",save_file[5,3])
    CGS6LR = save_file[5,4]
    cv2.setTrackbarPos("C6 GS LR" ,"Settings Window (Colors 4 - 6)",save_file[5,4])
    

reload()
#called to generate the inital image with the default settings

key = ""
while key != ord('e') and key != 27:
    key = cv2.waitKey(0)
    if key == ord('s'):
        scale_percent = (1/rS * 100) + 0.00008
        width_grey = int(output.shape[1] * scale_percent)
        height_grey = int(output.shape[0] * scale_percent)
        dim_grey = (width_grey, height_grey)
        output = cv2.resize(output, dim_grey, interpolation = cv2.INTER_AREA)
        tkinter.Tk().withdraw() # prevents an empty tkinter window from appearing
        savename = filedialog.asksaveasfilename(filetypes = [(".jpg",".jpg"),(".png",".png"),(".jpeg",".jpeg"),(".jfif",".jfif")],defaultextension  = ".jpg",title = "Save Image",initialdir = os.path.join("C:\\","Users",os.path.expanduser("~"),"Downloads"))
        if(savename != ""):
            cv2.imwrite(savename,output)
    elif(key == ord('w')):
        savename = filedialog.asksaveasfilename(defaultextension  = ".png",title = "Save Preset",initialdir = os.path.join(dir,"saves"))
        if(savename != ""):
            cv2.imwrite(savename, numpy.array(saveImage(),dtype = "uint8"))
cv2.destroyAllWindows()
process.terminate()

