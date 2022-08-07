import cv2
import numpy as np

def movie_save():
    i = 0
    img_1 = cv2.imread("C:/touhou/train/7/" + str(i) + ".png")
    print(img_1.shape)
    img_2 = cv2.imread("C:/touhou/eval_img/7/" + str(i) + ".png")

    mergeimg = np.hstack((img_1, img_2))
    sh = mergeimg.shape
    print(sh)

    size = (sh[1], sh[0])
    fourcc = cv2.VideoWriter_fourcc("m", "p" , "4", "v")
    save = cv2.VideoWriter("sample.mp4", fourcc, 5.0, size)

    for i in range(100):
        img_1 = cv2.imread("C:/touhou/train/7/" + str(i) + ".png")
        img_2 = cv2.imread("C:/touhou/eval_img/7/" + str(i) + ".png")

        mergeimg = np.hstack((img_1, img_2))
        mergeimg = cv2.resize(mergeimg, (sh[1], sh[0]))
        save.write(mergeimg)

    save.release()

if __name__ == "__main__":
    movie_save()