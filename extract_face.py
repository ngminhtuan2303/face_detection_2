#------------------------------
#extract face with alignment
from retinaface import RetinaFace
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np


img_paths = ["data/img11.jpg", "data/img6.jpg"]
rotate_angles = [0, 30, 45, 60, 90, -30, -45, -60, -90]

for img_path in img_paths:

    #resp = RetinaFace.extract_faces(img_path = img_path, align = True)
    img = cv2.imread(img_path)
    img_base = img.copy()

    for angle in rotate_angles:
        print(f"rotating {img_path} to {angle} degrees")
        img = img_base.copy()
        img = Image.fromarray(img)
        img = np.array(img.rotate(angle))

        faces = RetinaFace.extract_faces(img_path = img, align = True)

        for face in faces:
            plt.imshow(face)
            plt.axis('off')
            plt.show()
            cv2.imwrite('outputs/'+img_path.split("/")[1], face[:, :, ::-1])