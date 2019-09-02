""" Extracts frames from a video and stores triplets for network consumption """

from absl import app
from absl import flags
import numpy as np
import cv2
from collections import deque
from os import path

flags.DEFINE_string('input_file', None, 'Input video file')
flags.DEFINE_string('output_dir', None, 'Directory to store predictions.')
flags.DEFINE_integer('img_height', 128, 'Input frame height.')
flags.DEFINE_integer('img_width', 416, 'Input frame width.')

FLAGS = flags.FLAGS
flags.mark_flag_as_required('input_file')
flags.mark_flag_as_required('output_dir')


def main(_):
    cap = cv2.VideoCapture(FLAGS.input_file)
    queue = deque(maxlen=3)
    ctr = 0
    while cap.isOpened():
        _, frame = cap.read()
        frame = cv2.resize(frame, (FLAGS.img_width, FLAGS.img_height))
        queue.append(frame)
        if len(queue) == 3:
            triplet = np.zeros((FLAGS.img_height, FLAGS.img_width * 3, 3), dtype=np.uint8)
            for i in range(3):
                img = queue[i]
                min_x = i*FLAGS.img_width
                max_x = min_x + FLAGS.img_width
                triplet[:, min_x:max_x, :] = img
            filename = path.join(FLAGS.output_dir, 'triplet_%08d.png' % ctr)
            cv2.imwrite(filename, triplet)
            ctr += 1
            queue.popleft()
    cap.release()

if __name__ == '__main__':
  app.run(main)