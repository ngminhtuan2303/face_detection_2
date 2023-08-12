from retinaface import RetinaFace
import matplotlib.pyplot as plt
import cv2
# from PIL import Image
import numpy as np

# img_path = "data/img6.jpg"
img_paths = ["data/img11.jpg", "data/img6.jpg"]



#print(resp)

def int_tuple(t):
    return tuple(int(x) for x in t)

for img_path in img_paths:
    img = cv2.imread(img_path)
    resp = RetinaFace.detect_faces(img_path, threshold = 0.1)

    for key in resp:
        identity = resp[key]

        #---------------------
        confidence = identity["score"]

        rectangle_color = (255, 255, 255)

        landmarks = identity["landmarks"]
        diameter = 1
        cv2.circle(img, int_tuple(landmarks["left_eye"]), diameter, (0, 0, 255), 5)
        cv2.circle(img, int_tuple(landmarks["right_eye"]), diameter, (0, 0, 255), 5)
        cv2.circle(img, int_tuple(landmarks["nose"]), diameter, (0, 0, 255), 5)
        cv2.circle(img, int_tuple(landmarks["mouth_left"]), diameter, (0, 0, 255), 5)
        cv2.circle(img, int_tuple(landmarks["mouth_right"]), diameter, (0, 0, 255), 5)

        facial_area = identity["facial_area"]

        cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), rectangle_color, 5)
        facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
        plt.imshow(facial_img[:, :, ::-1])

    plt.imshow(img[:, :, ::-1])
    plt.axis('off')
    plt.show()
    cv2.imwrite('outputs/'+img_path.split("/")[1], img)

#------------------------------
#extract face with alignment




# faces = RetinaFace.extract_faces(img_path, align = True)

# for face in faces:
#     plt.imshow(face)
#     plt.axis('off')
#     plt.show()
#     cv2.imwrite('outputs/'+img_path.split("/")[1], face[:, :, ::-1])





# Img1
### Detect face + landmark by RetinaFace
### Use landmark to alignment face : Dựa vào 5 landmarks trải phẳng mặt ra
### Face extraction (ArcFace), tìm example load model = pytorch => infer => vector 512, 1024

# Img2
### Detect face + landmark by RetinaFace
### Use landmark to alignment face : Dựa vào 5 landmarks trải phẳng mặt ra
### Face extraction (ArcFace), tìm example load model = pytorch => infer => vector 512, 1024

# Cosine distance feature_img1 vs feature_img2 > score > threshold > match / non-match

