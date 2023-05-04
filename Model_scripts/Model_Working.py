from Model_Deps import *

def main_test():
    model = MyModel()

    cap = cv2.VideoCapture("../DATA/video/1.mp4")
    while(cap.isOpened()):
        success, img = cap.read()
        if success == True:
            img = model.image_segmentation(img)
            cv2.imshow("Segmented Image", img)

            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def main_film():
    model = MyModel()

    cap = cv2.VideoCapture("../DATA/video/7.mp4")
    result = cv2.VideoWriter('../ready_vids/after_optim/vid7_optim.avi', cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (1536, 864))
    while(cap.isOpened()):
        success, img = cap.read()
        if success == True:
            img = model.image_segmentation(img)
            cv2.imshow("Segmented Image", img)
            result.write(img)

            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
        else:
            break
    cap.release()
    result.release()
    cv2.destroyAllWindows()
if __name__=="__main__":
    main_film()