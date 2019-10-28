import argparse
import flow_methods as fm
import scipy.misc

"""this file applies the optical flow for a given stylized fram from a video.
    Call this script for each stylized frame that is computed to warp it.
"""



parser = argparse.ArgumentParser(description='Flow Configuration')
parser.add_argument('--name', dest='name', default='vid1', type=str, help='video name without filename extension')
parser.add_argument('--style', dest='style', default='vid1', type=str, help='video style without filename extension')
parser.add_argument('--toPos', dest='toPos', default='2', type=int, help='second frame number')
parser.add_argument('--imsize', nargs='+', dest='imsize',  default=[100,100], type=int, help='resulution of images hight width')

parser: argparse.Namespace = parser.parse_args()

name = parser.name
style = parser.style
toPos = parser.toPos
imsize = parser.imsize

flow = fm.read_flow("videos/{}flow/forward_{:04d}_{:04d}.flo".format(name, toPos-1, toPos))

if len(imsize) > 1:
    stylized_name = "stylized-{}-{}-{}_{}".format(name, style, imsize[0],imsize[1])
else:
    print("abnormal style loading, only one size argument")
    stylized_name = "stylized-{}-{}-{}".format(name, style, imsize[0])

img = scipy.misc.imread("output/video/{}/{}-v{:04d}.png".format(name, stylized_name, toPos-1))

print(img.shape, flow.shape)
warped_image = fm.warp_image(img, flow)
save_name = "videos/{}flow/initImage_{:04d}.png".format(name, toPos)
scipy.misc.imsave(save_name, warped_image)

#for comparison
content_path = "videos/{}/frame_{:04d}.ppm".format(name, toPos - 1)
content_img = scipy.misc.imread(content_path)
warped_content = fm.warp_image(content_img, flow)
save_name = "videos/{}flow/content_flow_{:04d}.png".format(name, toPos)
scipy.misc.imsave(save_name, warped_content)
