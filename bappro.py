import cv2
import numpy as np
import os

frames_folder = 'c:/Users/Kyasarou/Downloads/bad-apple-frames-main/frames'
os.makedirs('points', exist_ok=True)
total_frames = len(os.listdir(frames_folder))

with open('points/all_points.txt', 'w') as all_points_file:
    for i, frame in enumerate(os.listdir(frames_folder), start=1):
        img = cv2.imread(os.path.join(frames_folder, frame), cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        points = []
        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            for point in approx:
                x, y = point[0]
                points.append((x, y))
        
        all_points_file.write(f'Frame {i}:\n')
        for x, y in points:
            all_points_file.write(f'{x},{y}\n')
        
        print(f'Processed frame {i} of {total_frames}')

