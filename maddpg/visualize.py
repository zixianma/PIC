
import argparse
from ckpt_plot.plot_curve import read_csv
from utils import create_video


parser = argparse.ArgumentParser(description='Eval agent visualization')
parser.add_argument('--video_file', type=str, default='videos/video.mp4',
                    help='name of the environment to run')
parser.add_argument('--frame_data_file', type=str, default='frame_data.csv',
                    help='the path to the file storing frame data')
args = parser.parse_args()

data = read_csv(args.frame_data_file)
num_render = len(data['frames'])
for i in range(num_render):
    create_video(data['frames'], args.video_file.replace(
                        '.mp4', '_%d.mp4' % i))