import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import pandas as pd
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
    check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
    increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# For SORT tracking
import skimage
from sort import *

# ............................... Tracker Functions ............................
""" Random created palette"""
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

# right turn
# 1
area0_pointA = (520, 350)
area0_pointB = (538, 350)
area0_pointC = (510, 370)
area0_pointD = (525, 370)


# 1
area1_pointA = (538, 350)
area1_pointB = (565, 350)
area1_pointC = (525, 370)
area1_pointD = (555, 370)

# 2
area2_pointA = (565, 350)
area2_pointB = (590, 350)
area2_pointC = (555, 370)
area2_pointD = (580, 370)

# 3
area3_pointA = (595, 350)
area3_pointB = (620, 350)
area3_pointC = (585, 370)
area3_pointD = (610, 370)

# 4
area4_pointA = (620, 350)
area4_pointB = (648, 350)
area4_pointC = (610, 370)
area4_pointD = (638, 370)
#
# 5
area5_pointA = (648, 350)
area5_pointB = (670, 350)
area5_pointC = (638, 370)
area5_pointD = (670, 370)

# 6
area6_pointA = (625, 640)
area6_pointB = (667, 641)
area6_pointC = (625, 665)
area6_pointD = (663, 665)


# 7
area7_pointA = (667, 640)
area7_pointB = (707, 641)
area7_pointC = (665, 665)
area7_pointD = (702, 668)

# 8
area8_pointA = (708, 638)
area8_pointB = (750, 642)
area8_pointC = (707, 667)
area8_pointD = (742, 666)

# 10
area9_pointA = (740, 679)
area9_pointB = (778, 679)
area9_pointC = (739, 697)
area9_pointD = (773, 695)

# 11
area10_pointA = (781, 608)
area10_pointB = (813, 606)
area10_pointC = (782, 631)
area10_pointD = (817, 633)


# 11
area11_pointA = (866, 525)
area11_pointB = (905, 525)
area11_pointC = (866, 550)
area11_pointD = (900, 550)

counting_0 = 0
modulo_counting_0 = 0
counting_1 = 0
modulo_counting_1 = 0
counting_2 = 0
modulo_counting_2 = 0
counting_3 = 0
modulo_counting_3 = 0
counting_4 = 0
modulo_counting_4 = 0
counting_5 = 0
modulo_counting_5 = 0
counting_6 = 0
modulo_counting_6 = 0
counting_7 = 0
modulo_counting_7 = 0
counting_8 = 0
modulo_counting_8 = 0
counting_9 = 0
modulo_counting_9 = 0
counting_10 = 0
modulo_counting_10 = 0
counting_11 =0
modulo_counting_11 = 0




"""" Calculates the relative bounding box from absolute pixel values. """
def count_vehicles(count_vehicle, counting, array_ids, modulo_counting):
    if count_vehicle == 0:
        counting = len(array_ids)
    else:
        if counting < 100:
            counting = len(array_ids)
        else:
            counting = modulo_counting + len(array_ids)
            if len(array_ids) % 100 == 0:
                modulo_counting += 100
                array_ids.clear()
    return counting, modulo_counting

def check_area(midpoint_x, midpoint_y, area_pointA, area_pointD, array_ids, id, label):

    # Check if the vehicle has crossed the line
    if (midpoint_x > area_pointA[0] and midpoint_x < area_pointD[0]) and (midpoint_y > area_pointA[1] and midpoint_y < area_pointD[1]):
        midpoint_color = (0, 0, 255)
        print('Kategori : ' + str(label))

        # Add vehicles counting
        if len(array_ids) > 0:
            if label not in array_ids:
                array_ids.append(label)
        else:
            array_ids.append(label)
    return array_ids


def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


"""Simple function that adds fixed color depending on the class"""


def compute_color_for_labels(label):
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)



def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0), array_ids_0 =[], array_ids_1=[], array_ids_2=[], array_ids_3=[], array_ids_4=[], array_ids_5=[], array_ids_6=[], array_ids_7=[], array_ids_8=[], array_ids_9=[], array_ids_10=[], array_ids_11=[]):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        data = (int((box[0] + box[2]) / 2), (int((box[1] + box[3]) / 2)))
        label = str(id) + ":" + names[cat]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 144, 30), 1)
        # cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
        # cv2.circle(img, data, 6, color,-1)

        # c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        midpoint_x = x1 + ((x2 - x1) / 2)
        midpoint_y = y1 + ((y2 - y1) / 2)
        center_point = (int(midpoint_x), int(midpoint_y))
        midpoint_color = (0, 255, 0)
        # vehicles total counting variables

        array_ids_0 = check_area(midpoint_x, midpoint_y, area0_pointA, area0_pointD, array_ids_0, id, label)
        array_ids_1 = check_area(midpoint_x, midpoint_y, area1_pointA, area1_pointD, array_ids_1, id, label)
        array_ids_2 = check_area(midpoint_x, midpoint_y, area2_pointA, area2_pointD, array_ids_2, id, label)
        array_ids_3 = check_area(midpoint_x, midpoint_y, area3_pointA, area3_pointD, array_ids_3, id, label)
        array_ids_4 = check_area(midpoint_x, midpoint_y, area4_pointA, area4_pointD, array_ids_4, id, label)
        array_ids_5 = check_area(midpoint_x, midpoint_y, area5_pointA, area5_pointD, array_ids_5, id, label)
        array_ids_6 = check_area(midpoint_x, midpoint_y, area6_pointA, area6_pointD, array_ids_6, id, label)
        array_ids_7 = check_area(midpoint_x, midpoint_y, area7_pointA, area7_pointD, array_ids_7, id, label)
        array_ids_8 = check_area(midpoint_x, midpoint_y, area8_pointA, area8_pointD, array_ids_8, id, label)
        array_ids_9 = check_area(midpoint_x, midpoint_y, area9_pointA, area9_pointD, array_ids_9, id, label)
        array_ids_10 = check_area(midpoint_x, midpoint_y, area10_pointA, area10_pointD, array_ids_10, id, label)
        array_ids_11 = check_area(midpoint_x, midpoint_y, area11_pointA, area11_pointD, array_ids_11, id, label)


        cv2.circle(img, center_point, radius=1, color=midpoint_color, thickness=2)

    return img


# ..............................................................................


def detect(save_img=False,counting_0 = 0,modulo_counting_0 = 0, counting_1=0, modulo_counting_1=0, counting_2=0, modulo_counting_2=0, counting_3=0, modulo_counting_3=0, counting_4=0, modulo_counting_4=0, counting_5=0, modulo_counting_5=0,counting_6=0, modulo_counting_6 =0, counting_7=0,modulo_counting_7=0,counting_8=0, modulo_counting_8=0, counting_9=0, modulo_counting_9=0,counting_10=0, modulo_counting_10=0,counting_11=0, modulo_counting_11=0, array_ids_0 = [], array_ids_1=[], array_ids_2=[], array_ids_3=[], array_ids_4=[], array_ids_5=[], array_ids_6=[], array_ids_7=[], array_ids_8=[], array_ids_9=[], array_ids_10=[], array_ids_11=[]):

    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # .... Initialize SORT ....
    # .........................
    sort_max_age = 5
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)
    # .........................
    # Directories

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    half = False

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    count_vehicle = 0

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]

            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # ..................USE TRACK FUNCTION....................
                # pass an empty array to sort
                dets_to_sort = np.empty((0, 6))

                # NOTE: We send in detected object class too
                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort,
                                              np.array([x1, y1, x2, y2, conf, detclass])))

                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks = sort_tracker.getTrackers()

                # print('Tracked Detections : ' + str(len(tracked_dets)))

                # loop over tracks
                '''
                for track in tracks:
                    # color = compute_color_for_labels(id)
                    #draw tracks

                    [cv2.line(im0, (int(track.centroidarr[i][0]),
                                    int(track.centroidarr[i][1])), 
                                    (int(track.centroidarr[i+1][0]),
                                    int(track.centroidarr[i+1][1])),
                                    (0,255,0), thickness=1) 
                                    for i,_ in  enumerate(track.centroidarr) 
                                        if i < len(track.centroidarr)-1 ] 
                '''

                # draw boxes for visualization
                if len(tracked_dets) > 0:
                    bbox_xyxy = tracked_dets[:, :4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    draw_boxes(im0, bbox_xyxy, identities, categories, names, array_ids_0 = array_ids_0, array_ids_1=array_ids_1, array_ids_2=array_ids_2, array_ids_3=array_ids_3, array_ids_4=array_ids_4, array_ids_5=array_ids_5, array_ids_6=array_ids_6, array_ids_7=array_ids_7, array_ids_8=array_ids_8, array_ids_9=array_ids_9, array_ids_10=array_ids_10, array_ids_11=array_ids_11)
                    # array_ids_5=array_ids_5, array_ids_6=array_ids_6)
                    print('Bbox xy count : ' + str(len(bbox_xyxy)))
                # ........................................................

            # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            cv2.line(im0, area0_pointA, area0_pointB, (0, 0, 0), 2)
            cv2.line(im0, area0_pointC, area0_pointD, (0, 0, 0), 2)

            cv2.line(im0, area1_pointA, area1_pointB, (255, 0, 0), 2)
            cv2.line(im0, area1_pointC, area1_pointD, (255, 0, 0), 2)

            cv2.line(im0, area2_pointA, area2_pointB, (0, 0, 255), 2)
            cv2.line(im0, area2_pointC, area2_pointD, (0, 0, 255), 2)

            cv2.line(im0, area3_pointA, area3_pointB, (255, 255, 0), 2)
            cv2.line(im0, area3_pointC, area3_pointD, (255, 255, 0), 2)

            cv2.line(im0, area4_pointA, area4_pointB, (0, 255, 255), 2)
            cv2.line(im0, area4_pointC, area4_pointD, (0, 255, 255), 2)

            cv2.line(im0, area5_pointA, area5_pointB, (0, 255, 0), 2)
            cv2.line(im0, area5_pointC, area5_pointD, (0, 255, 0), 2)

            cv2.line(im0, area6_pointA, area6_pointB, (0, 125, 0), 2)
            cv2.line(im0, area6_pointC, area6_pointD, (0, 125, 0), 2)

            cv2.line(im0, area7_pointA, area7_pointB, (125, 0, 0), 2)
            cv2.line(im0, area7_pointC, area7_pointD, (125, 0, 0), 2)

            cv2.line(im0, area8_pointA, area8_pointB, (0, 0, 125), 2)
            cv2.line(im0, area8_pointC, area8_pointD, (0, 0, 125), 2)

            cv2.line(im0, area9_pointA, area9_pointB, (125, 125, 0), 2)
            cv2.line(im0, area9_pointC, area9_pointD, (125, 125, 0), 2)

            cv2.line(im0, area10_pointA, area10_pointB, (0, 125, 125), 2)
            cv2.line(im0, area10_pointC, area10_pointD, (0, 125, 125), 2)

            cv2.line(im0, area11_pointA, area11_pointB, (125, 0, 125), 2)
            cv2.line(im0, area11_pointC, area11_pointD, (125, 0, 125), 2)



            color = (0, 255, 0)
            thickness = 1
            fontScale = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (160, 570)

            counting_0, modulo_counting_0 = count_vehicles(count_vehicle, counting_0, array_ids_0, modulo_counting_0)
            counting_1, modulo_counting_1 = count_vehicles(count_vehicle, counting_1, array_ids_1, modulo_counting_1)
            counting_2, modulo_counting_2 = count_vehicles(count_vehicle, counting_2, array_ids_2, modulo_counting_2)
            counting_3, modulo_counting_3 = count_vehicles(count_vehicle, counting_3, array_ids_3, modulo_counting_3)
            counting_4, modulo_counting_4 = count_vehicles(count_vehicle, counting_4, array_ids_4, modulo_counting_4)
            counting_5, modulo_counting_5 = count_vehicles(count_vehicle, counting_5, array_ids_5, modulo_counting_5)
            counting_6, modulo_counting_6 = count_vehicles(count_vehicle, counting_6, array_ids_6, modulo_counting_6)
            counting_7, modulo_counting_7 = count_vehicles(count_vehicle, counting_7, array_ids_7, modulo_counting_7)
            counting_8, modulo_counting_8 = count_vehicles(count_vehicle, counting_8, array_ids_8, modulo_counting_8)
            counting_9, modulo_counting_9 = count_vehicles(count_vehicle, counting_9, array_ids_9, modulo_counting_9)
            counting_10, modulo_counting_10 = count_vehicles(count_vehicle, counting_10, array_ids_10, modulo_counting_10)
            counting_11, modulo_counting_11 = count_vehicles(count_vehicle, counting_11, array_ids_11, modulo_counting_11)


            print(array_ids_6,array_ids_7, array_ids_8, array_ids_9, array_ids_10, array_ids_11)
            # print(array_ids_0, counting_1, counting_2, counting_3, counting_4,counting_5)

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    # print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

    df = pd.DataFrame({
        'List0': pd.Series(array_ids_0),
        'List1': pd.Series(array_ids_1),
        'List2': pd.Series(array_ids_2),
        'List3': pd.Series(array_ids_3),
        'List4': pd.Series(array_ids_4),
        'List5': pd.Series(array_ids_5),
        'List6': pd.Series(array_ids_6),
        'List7': pd.Series(array_ids_7),
        'List8': pd.Series(array_ids_8),
        'List9': pd.Series(array_ids_9),
        'List10': pd.Series(array_ids_10),
        'List11': pd.Series(array_ids_11),

    })

    # Save the DataFrame to a CSV file
    df.to_csv('tri_1_row.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/yolov7x2/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images/testssss.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.65, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='object_tracking', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)


    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['runs/train/yolov7x2/weights/best.pt']:
                detect(save_img=False,counting_0=counting_0,modulo_counting_0=modulo_counting_0, counting_1=counting_1, modulo_counting_1=modulo_counting_1, counting_2=counting_2, modulo_counting_2=modulo_counting_2, counting_3=counting_3, modulo_counting_3=modulo_counting_3, counting_4=counting_4, modulo_counting_4=modulo_counting_4, counting_5=counting_5, modulo_counting_5=modulo_counting_5, counting_6=counting_6, modulo_counting_6=modulo_counting_6, counting_7=counting_7, modulo_counting_7=modulo_counting_7, counting_8=counting_8, modulo_counting_8=modulo_counting_8, counting_9=counting_9, modulo_counting_9=modulo_counting_9, counting_10=counting_10, modulo_counting_10=modulo_counting_10, counting_11=counting_11, modulo_counting_11=modulo_counting_11)
                strip_optimizer(opt.weights)

        else:
            detect(save_img=False, counting_0=counting_0,modulo_counting_0=modulo_counting_0, counting_1=counting_1, modulo_counting_1=modulo_counting_1, counting_2=counting_2, modulo_counting_2=modulo_counting_2, counting_3=counting_3, modulo_counting_3=modulo_counting_3, counting_4=counting_4, modulo_counting_4=modulo_counting_4, counting_5=counting_5, modulo_counting_5=modulo_counting_5, counting_6=counting_6, modulo_counting_6=modulo_counting_6, counting_7=counting_7, modulo_counting_7=modulo_counting_7, counting_8=counting_8, modulo_counting_8=modulo_counting_8, counting_9=counting_9, modulo_counting_9=modulo_counting_9, counting_10=counting_10, modulo_counting_10=modulo_counting_10, counting_11=counting_11, modulo_counting_11=modulo_counting_11)