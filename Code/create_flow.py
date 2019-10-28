import cv2
import argparse
import flow_methods as fm

"""This script creates the optical flow for all the frames in a video.
    The forward and backwards flow are computed and saved.
    Call this script for a video that had been composed into its single frames
"""

parser = argparse.ArgumentParser(description='Flow Configuration')
parser.add_argument('--name', dest='name', default='vid1', type=str, help='video name without ending')

parser: argparse.Namespace = parser.parse_args()

name = parser.name


cap = cv2.VideoCapture("videos/{}/frame_%4d.ppm".format(name))
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

print("creates flow for {}.".format(name))
i = 0
while(1):

    i += 1
    ret, frame2 = cap.read()

    if ret is False:
        print("breaks {}".format(name))
        break
    else:
        if frame2.shape[0] < 20:
            print("end")
            break
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow1 = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow2 = cv2.calcOpticalFlowFarneback(next, prvs, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        print(flow1.shape, i)
        fm.write_flow("videos/{}flow/forward_{:04d}_{:04d}.flo".format(name, i, i+1), flow1)
        fm.write_flow("videos/{}flow/backward_{:04d}_{:04d}.flo".format(name, i+1, i), flow2)
        fm.write_consistency("videos/{}flow/consistency_{:04d}_{:04d}".format(name, i, i+1), flow1, flow2)
        prvs = next
        fm.write_flowimage(name, i, flow1)

cap.release()


