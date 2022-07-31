import cv2

for i in range(1, 173):
    img = cv2.imread("C:/touhou/data/7/bullet/" + str(i) + ".png")
    print(type(img))
    print(img.shape)

    #img = cv2.flip(img, 1)
    img = cv2.resize(img, dsize=None, fx=0.6, fy=0.6)
    cv2.imwrite("C:/touhou/data/7/bullet_tmp/" + str(i) + ".png", img)