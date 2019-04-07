from ObjectDetection import YOLO
import cv2
import numpy as np
from tracking import *

VIDEO_DIR = './VideoDataSets/1.mp4'

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    ih, iw = image.shape[:2]
    h, w = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = cv2.resize(image, (nw,nh), cv2.INTER_CUBIC)
    new_image = np.zeros((w, h, 3), np.uint8)
    new_image[:,:] = (128, 128, 128)
    new_image[(h-nh)//2:(h-nh)//2 +nh, (w-nw)//2:(w-nw)//2 + nw] = image
    return new_image

selection_dict = {'img': None, 'points selected': []}

def select_point(event, x, y, flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(selection_dict['img'],(x,y), 5, (0, 255, 0), -1)
        selection_dict['points selected'].append([x, y])


def select_quadrilateral_from(image):
    selection_dict['img'] = image
    cv2.namedWindow('selection frame')
    cv2.setMouseCallback('selection frame', select_point)

    while(1):
        cv2.imshow('selection frame', image)
        if cv2.waitKey(20) & 0xFF == 27:
            break
        if len(selection_dict['points selected']) >= 4:
            break

    cv2.destroyAllWindows()
    if len(selection_dict['points selected']) != 4:
        return -1

    selection_dict['points selected'].sort(key=lambda point: point[1])

    """
        After sorting with y coordinate as key, the first two points represent the top two 
        points of the quadrilateral, and the next two represent the bottom two.
    """

    if selection_dict['points selected'][0][0] > selection_dict['points selected'][1][0]:
        selection_dict['points selected'][0], selection_dict['points selected'][1] = \
        selection_dict['points selected'][1], selection_dict['points selected'][0]

    if selection_dict['points selected'][3][0] > selection_dict['points selected'][2][0]:
        selection_dict['points selected'][3], selection_dict['points selected'][2] = \
        selection_dict['points selected'][2], selection_dict['points selected'][3]

    selection_dict['points selected'] = np.array(selection_dict['points selected'], dtype=np.int32)
    return 1

if __name__ == '__main__':
    model_image_size = (608, 608)
    yolo = YOLO()

    vehicle_count = 0
    vehicles = []
    cap = cv2.VideoCapture(VIDEO_DIR)
    ret, image = cap.read()
    image = letterbox_image(image, tuple(reversed(model_image_size)))


    if select_quadrilateral_from(image) == -1:
        print("You must select 4 points")
        cap.release()
        yolo.session_close()
        exit(0)

    quad_as_contour = selection_dict['points selected'].reshape((-1, 1, 2))

    distance = int(input("Enter the length of the selected region in meters: "))
    avg_speed = 0

    while True:
        start = time.time()
        ret, image = cap.read()
        if image is None:
            break

        image = letterbox_image(image, tuple(reversed(model_image_size)))
        boxes = yolo.detect_image(image)

        """
        Here we need to track
        """
        selected_boxes = []
        for box in boxes:
            y_mid = (box[0] + box[1])//2
            x_mid = (box[2] + box[3])//2
            if cv2.pointPolygonTest(selection_dict['points selected'], (x_mid, y_mid), measureDist=False) >=0 :
                selected_boxes.append(box)

        new_vehicles = not_tracked(selected_boxes, vehicles, vehicle_count)
        vehicle_velocity_sum, deleted_count = update_or_deregister(selected_boxes, vehicles, distance)

        if deleted_count != 0:
            avg_speed = int(avg_speed*vehicle_count + vehicle_velocity_sum//deleted_count)//(vehicle_count + deleted_count)

        vehicle_count += len(new_vehicles)
        vehicles += new_vehicles

        for vehicle in vehicles:
            cv2.rectangle(image, (vehicle.left, vehicle.top), (vehicle.right, vehicle.bottom), (255, 0, 0), 2)
            cv2.putText(image, str(vehicle.id), ((vehicle.left+vehicle.right)//2 -1 , (vehicle.top+vehicle.bottom)//2 + 1),
                        cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255, 255, 0),
                        thickness=1)
        end = time.time()
        cv2.putText(image, "FPS: " + str(int(1 / (end - start))), (0, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, "Count: " + str(vehicle_count), (608//2 - 20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, "Avg speed: " + str(avg_speed), (480, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.polylines(image, [quad_as_contour], True, (0, 255, 0), thickness=2)
        cv2.imshow('frame', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    yolo.session_close()

    print('___________________________STATISTICS___________________________')
    print('Vehicle count: ', vehicle_count)
    print('Avg speed of vehicles: ', avg_speed)