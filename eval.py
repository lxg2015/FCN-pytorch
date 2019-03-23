import cv2
import numpy as np
from skimage import io, measure, draw


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images
    Args
        input_image: np.ndarray with 3 channels
        segmentation_mask: np.ndarray with 1 channels
        alpha: float value
        https://github.com/meetshah1995/pytorch-semseg/blob/801fb200547caa5b0d91b8dde56b837da029f746/ptsemseg/utils.py#L25
    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    
    # First create the image with alpha channel
    # rgba = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2RGBA)
    # Then assign the mask to the last channel of the image
    # rgba[:, :, 3] = alpha_data
    
    return blended


def seg_iou(pred_mask, pred_area, gt_label, gt_regions):
    '''Find the max iou with gt_label
    '''
    dice_max = 0.
    idx_max = 0
    num_gt = len(gt_regions)
    for j in range(1, num_gt + 1):
        gt_mask = gt_label == j
        inter = (gt_mask * pred_mask).sum()
        gt_area = gt_regions[j - 1].area
        dice = 2 * inter / (pred_area + gt_area)
        if dice > dice_max:
            dice_max = dice
            idx_max = j
    return dice_max, idx_max
    

def image_evaluate(pred_img, gt_img, dice_thresh=0.2, src_img=None):
    '''Compute recall and precision for one image
    Args
        pred_img: np.ndarray with 3 channels
        gt_img: np.ndarray with 3 channels
        dice_thresh: float value
        src_img: np.ndarray with 3 channels
    '''
    pred_label, num_pred = measure.label(pred_img[:, :, 0], return_num=True)
    pred_regions = measure.regionprops(pred_label)
    
    gt_label, num_gt = measure.label(gt_img[:, :, 0], return_num=True)
    gt_regions = measure.regionprops(gt_label)

    recall_list = np.zeros(num_pred)
    precision_list = np.zeros(num_gt)

    for i in range(1, num_pred + 1):
        pred_mask = pred_label == i
        pred_area = pred_regions[i - 1].area
        dice_max, idx_max = seg_iou(pred_mask, pred_area, gt_label, gt_regions)
        if dice_max > dice_thresh:
            if recall_list[i - 1] == 0:
                recall_list[i - 1] = 1
                precision_list[idx_max - 1] = 1
        
    precision = recall_list.sum() / num_pred
    recall = recall_list.sum() / num_gt
    print('precision: %.5f recall: %.5f' % (precision, recall))

    if src_img is None:
        return precision, recall

    # show pred result
    # for i in range(1, num_pred + 1):
    #     bbox = pred_regions[i - 1].bbox
    #     print('i', i, bbox)
    #     color = (0, 255, 255) if recall_list[i - 1] == 1 else (0, 255, 0)
    #     src_img = cv2.rectangle(src_img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color, 2)
    src_img = alpha_blend(src_img, pred_img)

    # show gt
    for i in range(1, num_gt + 1):
        bbox = gt_regions[i - 1].bbox
        color = (255, 0, 0) if precision_list[i - 1] == 1 else (0, 0, 255)
        src_img = cv2.rectangle(src_img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color, 2)
    # cv2.imwrite('./picture/0_test.jpg', src_img)
    return precision, recall, src_img


def eval(pred_file, gt_file, src_file):
    with open(pred_file) as f:
        pred_list = f.readlines()
    with open(gt_file) as f:
        gt_list = f.readlines()
    with open(src_file) as f:
        src_list = f.readlines()
    
    num_img = len(pred_list)
    precision_list = []
    recall_list = []

    for i in range(num_img):
        print(i, end=' ')
        pred = cv2.imread(pred_list[i].strip())
        gt = cv2.imread(gt_list[i].strip())
        src = cv2.imread(src_list[i].strip())
        precision, recall, src_img = image_evaluate(pred, gt, 0.2, src)
        precision_list.append(precision)
        recall_list.append(recall)
        cv2.imwrite('./picture/%d.jpg' % i, src_img)

    precision = sum(precision_list) / num_img
    recall = sum(recall_list) / num_img
    print('mean precision: %.5f mean recall: %.5f' % (precision, recall))


def test():
    pred = cv2.imread('./picture/0_out.png')
    gt = cv2.imread('./picture/0_label.png')
    src = cv2.imread('./picture/0_src.jpg')    
    image_evaluate(pred, gt, 0.2, src)


if __name__ == '__main__':
    # test()
    pred_file = './picture/pred.txt'
    gt_file = './picture/gt.txt'
    src_file = './picture/src.txt'
    eval(pred_file, gt_file, src_file)
