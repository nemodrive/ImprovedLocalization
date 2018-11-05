import time
import numpy as np
import cv2
from PIL import Image
import numpy as np
from skimage.transform import rotate

NO_LOOPS = 1000

a = np.random.rand(1000 * 1000)
a = a.reshape((1000, 1000)) * 255
a = a.astype(np.uint8)

im = Image.fromarray(a)

ima = a / 255.0

rows, cols = a.shape
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)

t = []

for i in range(NO_LOOPS):
    st = time.time()

    # # PIL
    # im.rotate(10, Image.BICUBIC, expand=True)
    # # --

    # CV2
    dst = cv2.warpAffine(a,M,(cols,rows))
    # --
    #
    # # SKI
    # rotate(ima, 10)
    # # --
    #
    t.append(time.time()-st)

np.mean(t)
