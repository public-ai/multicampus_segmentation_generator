import cv2
import numpy as np
from tqdm import tqdm
import tensorflow.python.keras as keras


def image_contour_info(image):
    # color 2 gray
    ret, mask = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    binary_mask = np.where(mask == 0, 0, 255)

    # find contours
    copy_mask = binary_mask.copy().astype('uint8')
    contours, hierachy = cv2.findContours(copy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # fill polygons
    x, y, w, h = cv2.boundingRect(copy_mask)

    # 4D -> 2D
    contours = np.squeeze(contours)
    return copy_mask, contours, (x, y, w, h)


def generate_detection_images(train_imgs, train_labs, max_object, min_size: tuple, max_size: tuple):
    rand_bg_h = np.random.randint(min_size[0], max_size[0] + 1)
    rand_bg_w = np.random.randint(min_size[1], max_size[1] + 1)
    bg = np.zeros([rand_bg_h, rand_bg_w])
    color_bg = np.zeros([rand_bg_h, rand_bg_w, 3], dtype=np.int)

    ret_contours = []
    ret_bboxes = []
    ret_labels = []
    for i in range(max_object):

        rand_train_ind = np.random.randint(0, len(train_labs) - 1)

        # select patch image from training set
        train_img = train_imgs[rand_train_ind]
        train_lab = train_labs[rand_train_ind]
        img_h, img_w = train_img.shape

        # generate random image coordinates
        rand_patch_x = np.random.randint(0, rand_bg_w - img_w - 1)
        rand_patch_y = np.random.randint(0, rand_bg_h - img_h - 1)
        assert (rand_patch_x + img_w <= rand_bg_w) & \
               (rand_patch_y + img_h <= rand_bg_h)

        # patch image to background
        copy_mask, contours, (x, y, w, h) = image_contour_info(train_img)

        # 이렇게 하는 이유는 Contour 각 하나의 이미지에서 여러개 나오는 경우가 있다.
        # 이렇게 contour 가 여러개 나오는 경우는 그냥 pass 한다.
        # contour 가 하나만 나오는 경우만 적용한다.
        if len(np.shape(contours)) != 2:
            continue

        # mapping patch contours coordinates to background image
        assign_contours = contours + np.array([rand_patch_x, rand_patch_y])

        # mapping bboxes coordinates to background image
        assign_bbox = (rand_patch_x + x, rand_patch_y + y,
                       rand_patch_x + w + x, rand_patch_y + y + h)

        # extract patch image from background
        patch_mask = bg[rand_patch_y: rand_patch_y + img_h,
                     rand_patch_x: rand_patch_x + img_w].copy()

        # draw images
        # overlay check 구문
        # and 구문을 통해 겹치는게 하나라도 있으면 pass 한다.
        if np.sum((patch_mask != 0) & (copy_mask != 0)) == 0:

            # 이미지가 덮어 씌어진다. 덮어 씌어지는걸 방지하는 코드
            or_mask = (patch_mask != copy_mask)

            # gray 2 color
            color_or_masks = np.stack([or_mask, or_mask, or_mask], axis=-1)
            color_or_masks = np.asarray(color_or_masks != 0, dtype=np.int)

            # red channels
            rand_color_ind = np.random.randint(0, 255)
            color_or_masks[:, :, 0] = color_or_masks[:, :, 0] * rand_color_ind

            # green channels
            rand_color_ind = np.random.randint(0, 255)
            color_or_masks[:, :, 1] = color_or_masks[:, :, 1] * rand_color_ind

            # blue channels
            rand_color_ind = np.random.randint(0, 255)
            color_or_masks[:, :, 2] = color_or_masks[:, :, 2] * rand_color_ind

            # 백그라운드 이미지에 Patch 이미지 붙이기
            bg[rand_patch_y:rand_patch_y + img_h,
            rand_patch_x:rand_patch_x + img_w] = or_mask

            color_bg[rand_patch_y:rand_patch_y + img_h,
            rand_patch_x:rand_patch_x + img_w, :] = color_or_masks

            # append element to list
            if len(assign_contours.shape) == 2:
                ret_contours.append(assign_contours)
                ret_bboxes.append(assign_bbox)
                ret_labels.append(train_lab)

    assert (len(ret_labels) == len(ret_contours) == len(ret_bboxes))
    return color_bg, bg, ret_labels, ret_contours, ret_bboxes


def generate_segmentation_labels(bg_h, bg_w, contours, labels):
    global_bg = []
    coordinates = []

    for ind, label in enumerate(labels):
        # label 하나당 하나의 bg 에다가 patch 함
        local_bg = np.zeros([bg_h, bg_w])
        contour = contours[ind]
        #
        try:
            cv2.polylines(local_bg, [contour], True, (1, 0, 0), 1)
            cv2.fillPoly(local_bg, [contour], (1, 0, 0))
            x, y, w, h = cv2.boundingRect(contour)
        except:
            print(contour.shape)
        #
        local_bg = local_bg * (label + 1)

        coordinates.append([x, y, w, h])
        global_bg.append(local_bg)

    # 모든 bg 을 모아서 하나의
    global_bg = np.stack(global_bg, axis=-1)
    global_bg = np.sum(global_bg, axis=-1)
    global_bg = np.asarray(global_bg, np.int)
    return global_bg


def cls2onehot_2d(cls_2d, depth, offset=0):
    """
    clsonehot_2d([[1,2], [3,4]], 4, offset=-1)

    >>>
    array([[[1., 0., 0., 0.],
        [0., 1., 0., 0.]],

       [[0., 0., 1., 0.],
        [0., 0., 0., 1.]]])
    """
    h, w = cls_2d.shape
    cls_numbers = cls_2d.reshape(-1)
    cls_numbers = cls_numbers + offset

    onehot_vector = np.zeros([len(cls_numbers), depth])
    for ind, num in enumerate(cls_numbers):
        onehot_vector[ind, num] = 1
    return onehot_vector.reshape([h, w, -1])


def fasion_mnist_generator(n_train, n_test):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    min_h, max_h = 128, 128
    min_w, max_w = 128, 128
    n_classes = 11
    max_item = 10

    # 데이터셋을 생성합니다.
    ret_list = []
    for images, labels, n_samples in [(train_images, train_labels, n_train), (test_images, test_labels, n_test)]:
        color_bgs = []
        label_masks = []

        for i in tqdm(range(n_samples)):

            # generate images
            color_bg, bg, ret_labels, ret_contours, ret_bboxes = \
                generate_detection_images(images, labels, max_item, (min_h, min_w), (max_h, max_w))

            # generate labels
            label_mask = generate_segmentation_labels(min_h, min_w, ret_contours,
                                                      ret_labels)

            # label shape 을 class 로 부터 onehot 으로 변경합니다.
            label_mask_onehot = cls2onehot_2d(label_mask, n_classes)

            # append to list
            color_bgs.append(color_bg)
            label_masks.append(label_mask_onehot)

        color_bgs = np.asarray(color_bgs)
        label_masks = np.asarray(label_masks)
        ret_list.append((color_bgs, label_masks))
    return ret_list


if __name__ == '__main__':
    (train_xs, train_ys), (test_xs, test_ys) = fasion_mnist_generator(10, 10)
    print(train_xs.shape, train_ys.shape, test_xs.shape, test_ys.shape)

