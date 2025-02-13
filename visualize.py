from os.path import join
import os
import sys

import cv2
import numpy as np

# TRACKER_NAME = 'output_ua-detrac_unfreeze2'
# TRACKER_NAME = 'output_UA_DETRAC_ema'
TRACKER_NAME = 'quantyolov8_4w4a_mot15_cl0_bkp'
# TRACKER_NAME = 'output_lasteval_4w4a_mot17'
# SEQUENCES = ['MVI_40701', 'MVI_40771', 'MVI_40863']
SEQUENCES = 'all'
# SEQUENCES = ['TUD-Stadtmitte']
BENCHMARK = 'MOT15'

TRACKER_PATH = join('../sort/output', TRACKER_NAME)
DATASET_PATH = '/media/vision/1d6890f4-df75-4531-a044-f6d3d44d033d/Downloads/{}/train'.format(BENCHMARK)
# TRACKER_PATH = join('data/trackers/UA_DETRAC/', TRACKER_NAME)
# DATASET_PATH = '/media/vision/storage1/Datasets/UA_DETRAC/sorted/test'

def draw_bboxes(img, dets, color=(0, 0, 255), id_to_color=None, id_to_trajectory=None):
    # gt = dets.shape[-1] == 9
    # color = (0, 0, 255) if gt else (0, 255, 0)
    for det in dets:
        obj_id = det[1]
        if id_to_color is not None and obj_id not in id_to_color:
            id_to_color[obj_id] = (np.random.randint(256), np.random.randint(256), np.random.randint(256))
        xywh = det[2:6]
        xywh = [int(x) for x in xywh]
        conf = det[6]
        cls = det[7]
        # if cls == 4:
        if id_to_trajectory is not None:
            if obj_id not in id_to_trajectory:
                id_to_trajectory[obj_id] = []
            center = (xywh[0] + xywh[2]//2, xywh[1] + xywh[3]//2)
            id_to_trajectory[obj_id].append(center) 
            for point in id_to_trajectory[obj_id]:
                img = cv2.circle(img, point, radius=1, color=id_to_color[obj_id], thickness=2)
        img = cv2.rectangle(img, (xywh[0], xywh[1]), (xywh[0]+xywh[2], xywh[1]+xywh[3]), id_to_color[obj_id] if id_to_color else color, 2)
        font_scale = 0.5
        line_thickness = 1
        # print('HELLO')
        img = cv2.putText(img, str(det[7:]),
                        (xywh[0], xywh[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 255, 0),
                        line_thickness,
                        cv2.LINE_AA)


for seqname in os.listdir(DATASET_PATH):
    if SEQUENCES == 'all' or seqname in SEQUENCES:
        tracker_results_txt = join(TRACKER_PATH, 'data', seqname + '.txt')
        gt_txt = join(DATASET_PATH, seqname, 'gt', 'gt.txt')
        imgs_path = join(DATASET_PATH, seqname, 'img1')

        seq_tracks = np.loadtxt(tracker_results_txt, delimiter=',', dtype=float)
        seq_gt = np.loadtxt(gt_txt, delimiter=',', dtype=float)

        # print('t:', seq_tracks.shape)
        # print(seq_gt.shape)
        id_to_color = {}
        id_to_trajectory = {}

        imgnames = os.listdir(imgs_path)
        imgnames.sort()
        for frame_idx, imgname in enumerate(imgnames):
            frame_idx += 1
            imgpath = join(imgs_path, imgname)
            print(frame_idx, imgname)
            frame_tracks = seq_tracks[seq_tracks[:, 0] == frame_idx]
            # print('frame tracks:', frame_tracks)
            frame_gt = seq_gt[seq_gt[:, 0] == frame_idx]
            
            img = cv2.imread(imgpath)
            orig_img = img.copy()
            # draw_bboxes(img, frame_gt, (0, 0, 255))
            draw_bboxes(img, frame_tracks, color=(0, 255, 0), id_to_color=id_to_color, id_to_trajectory=id_to_trajectory)    

            cv2.imshow(seqname, img)
            key = cv2.waitKey(0)
            if key == ord('s'):
                cv2.destroyAllWindows()
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                sys.exit()
                
        cv2.destroyAllWindows()
            