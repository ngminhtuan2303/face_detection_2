from arcface.ArcFace import ArcFace
import matplotlib.pyplot as plt
import cv2
from retinaface import RetinaFace


img_paths = ["data/img11.jpg", "data/img6.jpg"]



for img_path in img_paths:
    img = cv2.imread(img_path)
    faces = RetinaFace.extract_faces(img_path = img, align = True)
    for face in faces:
        plt.imshow(face)
        plt.axis('off')
        plt.show()
        cv2.imwrite('outputs/'+img_path.split("/")[1], face[:, :, ::-1])


face_rec = ArcFace()
emb1 = face_rec.calc_emb("outputs/img11.jpg")
print("numpy",emb1)
emb2 = face_rec.calc_emb("outputs/img6.jpg")

# print(emb1)
dist = face_rec.get_distance_embeddings(emb1, emb2)
# assert dist > 0
# assert dist < 1.5
print(dist)

#Face search ::::: milvus