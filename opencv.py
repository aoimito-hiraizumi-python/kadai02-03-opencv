import cv2
from PIL import Image

original_face_img = "images/face_img.png"
original_eye_img = "images/eye_img.png"

def find_left_eye():
    cascade_file = "haarcascade_lefteye_2splits.xml"
    cascade = cv2.CascadeClassifier(cascade_file)
    img = cv2.imread(original_face_img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    left_eye_list = cascade.detectMultiScale(img_gray, minNeighbors=5)
    return left_eye_list

left_eye_list = find_left_eye()
print(left_eye_list)

def find_right_eye():
    cascade_file = "haarcascade_righteye_2splits.xml"
    cascade = cv2.CascadeClassifier(cascade_file)
    img = cv2.imread(original_face_img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    right_eye_list = cascade.detectMultiScale(img_gray, minNeighbors=11)
    return right_eye_list

right_eye_list = find_right_eye()
print(right_eye_list)

face_name_list = left_eye_list

def paste_img(face_name_list):
    x = face_name_list[0][0]
    y = face_name_list[0][1]
    w = face_name_list[0][2]
    h = face_name_list[0][3]
    print(x, y, w, h)

    face_img = Image.open(original_face_img)
    eye_img = Image.open(original_eye_img)

    new_eye_img = eye_img.resize((w, h))
    new_eye_img.save("images/resized_mask_img.png")

    face_img.paste(new_eye_img, (x, y), new_eye_img.split()[3])

    face_img.save("images/pasted_face.img.png")

paste_img(face_name_list)

face_name_list = right_eye_list
original_face_img = "images/pasted_face.img.png"
paste_img(face_name_list)

img = Image.open("images/pasted_face.img.png")
img.show()


