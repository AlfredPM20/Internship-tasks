import pygame
import sys
import numpy as np
from keras.models import load_model
import cv2

# Initialize pygame
pygame.init()

# Constants
WINDOWSIZEX = 600
WINDOWSIZEY = 400
BOUNDRYINC = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Load the trained model
MODEL = load_model("handwritten_digit_model.h5")

# Labels for digits
LABELS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

# Initialize screen
screen = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Handwritten Digit Recognition")

# Fonts
FONT = pygame.font.Font(None, 18)

# Variables
iswriting = False
number_xcord = []
number_ycord = []
image_cnt = 1
PREDICT = True

# Main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(screen, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)
        if event.type == pygame.MOUSEBUTTONDOWN:
            iswriting = True
        if event.type == pygame.MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)
            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDRYINC, 0), min(WINDOWSIZEX, number_xcord[-1] + BOUNDRYINC)
            rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDRYINC, 0), min(WINDOWSIZEY, number_ycord[-1] + BOUNDRYINC)
            number_xcord = []
            number_ycord = []
            img_array = np.array(pygame.PixelArray(screen))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)
            if PREDICT:
                image = cv2.resize(img_array, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28)) / 255
                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])
                textSurface = FONT.render(label, True, RED)
                textRecObj = textSurface.get_rect()
                textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_y
                screen.blit(textSurface, textRecObj)
        if event.type == pygame.KEYDOWN:
            if event.unicode == "n":
                screen.fill(BLACK)
    pygame.display.update()
