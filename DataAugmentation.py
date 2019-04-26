import os
from PIL import Image
import random

batches = os.listdir("Data/Raw")
# background_use_num = 1
# head_use_num = 1
size = 64
x = 0
y = 0
im_num = 0


# Use https://online.photoscissors.com/ to cut out head

def generating_waldos(use_background=True):
    background_use_num = 2
    head_use_num = 2
    size = 64
    im_num = 0
    # Getting head
    for head_batch in os.listdir("Data/Cleaning/OnlyWaldoHeads"):
        for head_name in os.listdir("Data/Cleaning/OnlyWaldoHeads/" + head_batch):
            for _ in range(head_use_num):
                for back_batch in os.listdir("Data/Cleaning/ClearedWaldo"):
                    for back_name in os.listdir("Data/Cleaning/ClearedWaldo/" + back_batch):
                        for _ in range(background_use_num):

                            # Rotate image
                            if random.randint(0, 9) < 5:
                                num = random.randint(-15, 15)
                                foreground = Image.open(
                                    "Data/Cleaning/OnlyWaldoHeads/" + head_batch + "/" + head_name).rotate(
                                    num)
                            else:
                                foreground = Image.open("Data/Cleaning/OnlyWaldoHeads/" + head_batch + "/" + head_name)

                            if random.randint(0, 9) < 7:
                                scale = random.uniform(0.8, 1.5)
                                w, h = foreground.size
                                foreground = foreground.resize((int(w * scale), int(h * scale)), Image.ANTIALIAS)
                            if use_background:
                                background = Image.open("Data/Cleaning/ClearedWaldo/" + back_batch + "/" + back_name)
                            else:
                                background = Image.open("Data/Cleaning/black_background.jpg")

                            # Crop background and place foreground on top
                            bck_w, bck_h = background.size
                            frg_w, frg_h = foreground.size

                            bck_x = random.randint(0, bck_w - size)
                            bck_y = random.randint(0, bck_h - size)
                            
                            frg_x = random.randint(0, 64 - frg_w)
                            frg_y = random.randint(0, 64 - frg_h)

                            cropped = background.crop((bck_x, bck_y, bck_x + size, bck_y + size))
                            cropped.save("Data/NotWaldo/n" + str(im_num) + ".png")
                            cropped.paste(foreground, (frg_x, frg_y), foreground)
                            cropped.save("Data/Waldo/" + str(im_num)+str(use_background) + ".png")
                            im_num += 1


generating_waldos(use_background=True)



