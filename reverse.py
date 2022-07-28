import cv2

for i in range(1, 188):
    img = cv2.imread("C:/touhou/data/7/enemy/" + str(i) + ".png")
    print(type(img))
    print(img.shape)

    img = cv2.flip(img, 1)
    cv2.imwrite("C:/touhou/data/7/enemy/" + str(i+187) + ".png", img)