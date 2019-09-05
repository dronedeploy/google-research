""" Extracts frames from a video and stores triplets for network consumption """

from absl import app
from absl import flags
import numpy as np
import cv2
from collections import deque
from os import path, walk, makedirs

flags.DEFINE_string('input_dir', None, 'Input video file')
flags.DEFINE_integer('img_height', 128, 'Input frame height.')
flags.DEFINE_integer('img_width', 416, 'Input frame width.')

FLAGS = flags.FLAGS
flags.mark_flag_as_required('input_dir')

def process_frames(input_dir, output_dir):
    queue = deque(maxlen=3)
    ctr = 0
    frame_idx = 0
    filename = path.join(input_dir, '%06d.jpg' % frame_idx)
    while path.isfile(filename):
        frame = cv2.imread(filename)
        frame = cv2.resize(frame, (FLAGS.img_width, FLAGS.img_height))
        queue.append(frame)
        if len(queue) == 3:
            triplet = np.zeros((FLAGS.img_height, FLAGS.img_width * 3, 3), dtype=np.uint8)
            for i in range(3):
                img = queue[i]
                min_x = i*FLAGS.img_width
                max_x = min_x + FLAGS.img_width
                triplet[:, min_x:max_x, :] = img
            filename = path.join(output_dir, 'triplet_%08d.png' % ctr)
            cv2.imwrite(filename, triplet)
            ctr += 1
            queue.popleft()
        frame_idx += 1

def main(_):
    for scene in next(walk(FLAGS.input_dir))[1]:
        input_dir = path.join(FLAGS.input_dir, scene, scene)
        output_dir = path.join(FLAGS.input_dir, scene, 'ml_depth_map_estimation_input')
        try:
            makedirs(output_dir)
        except Exception:
            continue
        process_frames(input_dir, output_dir)

if __name__ == '__main__':
  app.run(main)