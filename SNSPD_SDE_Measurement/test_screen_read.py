#%%
import os
import time
import logging
from pathlib import Path
from datetime import datetime
import pytesseract
import pyautogui
from PIL import Image

current_file_dir = Path(__file__).parent
logger = logging.getLogger(__name__)

def capture_screen_and_extract_text(x, y, width, height):
    logger.info("Take a screenshot of the area you want to capture...")
    
    # # Let the user select a region of the screen
    # screenshot = pyautogui.screenshot()
    # screenshot.save("full_screenshot.png")

    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    
    # Extract text from the screenshot
    text = pytesseract.image_to_string(screenshot)
    
    logger.info("\nExtracted Text:\n")
    print(text)
# %%
x=100
y=200
height=200
width=100
capture_screen_and_extract_text(x, y, width, height)
# %%
