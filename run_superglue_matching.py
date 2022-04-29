import os
import cv2 as cv

files = [2, 168, 214, 560, 1276, 1366, 1790, 2228, 2282, 2322, 2478, 2802, 2973]

match_threshold = 0.3
img_width = 1441
img_height = 400
imgs = 10

for i in range(len(files)):
    # Generate .txt file containing image pairs for matching
    txt_file = open(f'sintefmatching/img{files[i]}.txt', 'w')   
    for k in range(imgs):
        txt_file.write(f'left/l1200-{files[i] + k}.jpg right/r1200-{files[i] + k}.jpg\n')
    txt_file.close()

    input_dir = '../../test-images-sintef/1200rpm/'
    output_dir = f'sintefmatching/img{files[i]}/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    image_matches = f'sintefmatching/img{files[i]}.txt'

    # Run superglue matching
    os.system(f'./match_pairs.py \
    --input_dir {input_dir}/ \
    --input_pairs {image_matches} \
    --output_dir {output_dir} \
    --viz \
    --fast_viz \
    --resize {img_width} {img_height} \
    --match_threshold {match_threshold}')



