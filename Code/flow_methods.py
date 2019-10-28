import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import os

"""File with useful functions to apply optical flow.
"""



def write_flow(file_name, uv):
    """writes flow file

    Arguments:
        file_name {str} -- name of file
        uv {2d numpy array} -- the angle and magnitude of the flow
    """


    f = open(file_name, 'wb')

    assert(uv.ndim == 3)
    assert(uv.shape[2] == 2)
    u = uv[:, :, 0]
    v = uv[:, :, 1]
    assert(u.shape == v.shape)
    secondline, firstline = u.shape
    nBands = 2
    # write the header
    TAG = np.array([202021.25], np.float32)
    f.write(TAG)
    np.array(firstline).astype(np.int32).tofile(f)
    np.array(secondline).astype(np.int32).tofile(f)

    # arrange into matrix form
    tmp = np.zeros((secondline, firstline*nBands))
    tmp[:, np.arange(firstline)*2] = u
    tmp[:, np.arange(firstline)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)

    f.close()


def read_flow(file_name):
    """Reads flow file
    """

    print(file_name)
    assert type(file_name) is str
    assert os.path.isfile(file_name) is True
    assert file_name[-4:] == '.flo'

    TAG = np.array([202021.25], np.float32)
    f = open(file_name, 'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    data = np.fromfile(f, np.float32, count=2*w[0]*h[0])  # wenns nicht läuft mal dtype=np.float32 versuchen statt nur np.float32

    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    # forward: mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    return flow


def write_flowimage(contentname, number, flow):
    """Saves flow array in file

    Arguments:
        contentname {str} -- name of video
        number {str} -- frame number
        flow {np.array} -- flow in angular and magnitude
    """

    mag=flow[:,:,0]
    ang=flow[:,:,1]
    hsv = np.zeros((int(flow.shape[0]),flow.shape[1],3))
    hsv[...,1] = 255
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(np.uint8(hsv),cv2.COLOR_HSV2BGR)
    scipy.misc.imsave("output/video/{}/flowimage_{:04d}.png".format(contentname, number), bgr)


def warpimage(image, warp):
    """use warp_image instead
    """
    print(warp.shape[:2])
    height, width = warp.shape[:2]
    print(height, width)
    R2 = np.dstack(np.meshgrid(np.arange(width), np.arange(height)))

    pixel_map = R2 + warp
    # print(pixel_map)
    image = image.astype(np.float64)
    print(pixel_map.shape)
    print(image.shape)
    print(type(image[0,0,0]), type(pixel_map[0,0,0]))
    print(type(image), type(pixel_map))
    # print(image.type(),pixel_map.type())
    new_frame = cv2.remap(image, pixel_map[0], pixel_map[1], interpolation=0)
    plt.figure()
    plt.imshow(image)
    plt.figure()
    plt.imshow(new_frame)
    plt.show()


def warp_image2(im, flow):
    """Use optical flow to warp image to the next
    use warp_image instead

    Arguments:
        im: image to warp
        flow: optical flow

    Returns
        [np.uint8]: warped image

    """


    from scipy import interpolate
    image_height = im.shape[0]
    image_width = im.shape[1]
    flow_height = flow.shape[0]
    flow_width = flow.shape[1]
    n = image_height * image_width
    (iy, ix) = np.mgrid[0:image_height, 0:image_width]
    (fy, fx) = np.mgrid[0:flow_height, 0:flow_width]
    fx = fx.astype(np.float64)
    fy = fy.astype(np.float64)
    fx += flow[:, :, 0]
    fy += flow[:, :, 1]
    mask = np.logical_or(fx < 0, fx >= flow_width-1)
    mask = np.logical_or(mask, fy < 0)
    mask = np.logical_or(mask, fy >= flow_height-1)
    fx = np.minimum(np.maximum(fx, 0), flow_width)
    fy = np.minimum(np.maximum(fy, 0), flow_height)
    points = np.concatenate((ix.reshape(n, 1), iy.reshape(n, 1)), axis=1)
    xi = np.concatenate((fx.reshape(n, 1), fy.reshape(n, 1)), axis=1)
    warp = np.zeros((image_height, image_width, im.shape[2]))
    for i in range(im.shape[2]):
        channel = im[:, :, i]
        values = channel.reshape(n, 1)
        new_channel = interpolate.griddata(points, values, xi, method='cubic')
        new_channel = np.reshape(new_channel, [flow_height, flow_width])
        new_channel[mask] = im[:,:,i][mask]
        warp[:, :, i] = new_channel.astype(np.uint8)

    return warp.astype(np.uint8)


def warp_image(src, flow):
    """Warps the src image with the provided flow

    Arguments:
        src  -- Frame that should be warped
        flow  -- Optical flow, that the Frame should warped with
    """

    h, w, _ = flow.shape
    flow_map = np.zeros(flow.shape, dtype=np.float32)
    for y in range(h):
    flow_map[y,:,1] = float(y) + flow[y,:,1]
    for x in range(w):
    flow_map[:,x,0] = float(x) + flow[:,x,0]
    # remap pixels to optical flow
    dst = cv2.remap(
    src, flow_map[:,:,0], flow_map[:,:,1],
    interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
    return dst


def write_consistency(filename, flow1, flow2):
    """Calculates the consistency of two flows and saves the resulting mask in a .npy file.
    """

    f = open(filename, 'w')

    assert(flow1.ndim == 3)
    assert(flow1.shape[2] == 2)
    u = flow1[:, :, 0]
    v = flow1[:, :, 1]
    u2 = flow2[:, :, 0]
    v2 = flow2[:, :, 1]
    assert(u.shape == v.shape)
    secondline, firstline = u.shape
    stri="{} {} ".format(firstline,secondline)
    arr=np.array([firstline, secondline]).astype(np.int32)
    f.write(stri)
    mag=flow1[:,:,0]
    ang=flow1[:,:,1]
    mag2=flow2[:,:,0]
    ang2=flow2[:,:,1]
    hsv = np.zeros((int(secondline),int(firstline),3))
    hsv[...,1] = 255
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    hsv2 = np.zeros((int(secondline),int(firstline),3))
    hsv2[...,1] = 255
    hsv2[...,0] = ang2*180/np.pi/2
    hsv2[...,2] = cv2.normalize(mag2,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(np.uint8(hsv),cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    gray= gray.astype(np.uint8)
    bgr2 = cv2.cvtColor(np.uint8(hsv2),cv2.COLOR_HSV2BGR)
    gray2 = cv2.cvtColor(bgr2,cv2.COLOR_BGR2GRAY)
    gray2= gray2.astype(np.uint8)

    mask=np.zeros_like(gray)
    for j in range(np.size(gray,0)):
        for i in range(np.size(gray,1)):
            disoccl=(np.sqrt(gray[j,i]^2+gray2[j,i]^2))
            ornot=(0.01*(np.sqrt(gray[j,i]^2)+np.sqrt(gray[j,i]^2))+0.5)
            if disoccl> ornot:
                mask[j,i]=1
            else:
                mask[j,i]=0
    #print(mask.shape)
    np.save("{}.npy".format(filename), mask)
    print("{} 1ns in mask of {} elements.".format(np.sum(mask),np.size(mask)))


    f.close()