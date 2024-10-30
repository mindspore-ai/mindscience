# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===========================================================================
"""utils for map calculate"""
import glob
import json
import math
import operator
import os
import sys
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('Agg')

#     0,0 ------> x (width)
#      |
#      |  (Left,Top)
#      |      *_________
#      |      |         |
#             |         |
#      y      |_________|
#   (height)            *
#                 (Right,Bottom)


def log_average_miss_rate(precision, fp_cumsum, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.

        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = fp_cumsum / float(num_images)
    mr = 1 - precision

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi


def error(msg):
    """
     throw error and exit
    """
    print(msg)
    sys.exit(0)


def voc_ap(rec, prec):
    """
     Calculate the AP given the recall and precision array
        1st) We compute a version of the measured precision/recall curve with
             precision monotonically decreasing
        2nd) We compute the AP as the area under this curve by numerical integration.
    """
    rec.insert(0, 0.0)  # insert 0.0 at beginning of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at beginning of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    #  This part makes the precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    #  This part creates a list of indexes where the recall changes
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    #  The Average Precision (AP) is the area under the curve
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def file_lines_to_list(path):
    """
     Convert the lines of a file to a list
    """
    # open txt file lines to a list
    with open(path, encoding="utf-8") as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def draw_text_in_image(img, text, pos, color, line_width):
    """
     Draws text in image
    """
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1
    line_type = 1
    bottom_left_corner_of_text = pos
    cv2.putText(img, text,
                bottom_left_corner_of_text,
                font,
                font_scale,
                color,
                line_type)
    text_width, _ = cv2.getTextSize(text, font, font_scale, line_type)[0]
    return img, (line_width + text_width)


def adjust_axes(r, t, fig, axes):
    """
     Plot - adjust axes
    """
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    proportion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * proportion])


def draw_plot_func(dictionary, n_classes, window_title, plot_title,
                   x_label, output_path, to_show, plot_color, true_p_bar):
    """
     Draw plot using Matplotlib
    """
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(
        dictionary.items(),
        key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar != "":
        #  Special case to draw in:
        #     - green -> TP: True Positives (object detected and matches ground-truth)
        #     - red -> FP: False Positives (object detected but does not match ground-truth)
        #     - orange -> FN: False Negatives (object not detected but present in the ground-truth)
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(
            range(n_classes),
            fp_sorted,
            align='center',
            color='crimson',
            label='False Positive')
        plt.barh(
            range(n_classes),
            tp_sorted,
            align='center',
            color='forestgreen',
            label='True Positive',
            left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        #  Write number on side of bar
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            t = plt.text(
                val,
                i,
                tp_str_val,
                color='forestgreen',
                va='center',
                fontweight='bold')
            plt.text(
                val,
                i,
                fp_str_val,
                color='crimson',
                va='center',
                fontweight='bold')
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        #  Write number on side of bar
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)  # add a space before
            if val < 1.0:
                str_val = f" {val:.2f}"
            t = plt.text(
                val,
                i,
                str_val,
                color=plot_color,
                va='center',
                fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    #  Re-scale height accordingly
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4)  # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15  # in percentage of the figure height
    bottom_margin = 0.05  # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()


def get_map(minoverlap, draw_plot, score_threhold=0.5, path='./map_out'):
    """get map"""
    gt_path = os.path.join(path, 'ground-truth')
    dr_path = os.path.join(path, 'detection-results')
    temp_files_path = os.path.join(path, '.temp_files')
    result_files_path = os.path.join(path, 'results')

    def load_bounding_boxes(dr_files_list, gt_classes, score_threhold):
        bounding_boxes_per_class = {class_name: [] for class_name in gt_classes}
        for txt_file in dr_files_list:
            file_id = os.path.basename(txt_file).split(".txt", 1)[0]
            for line in file_lines_to_list(txt_file):
                tmp_class_name, confidence, left, top, right, bottom = parse_detection_line(line)
                if tmp_class_name in bounding_boxes_per_class and float(confidence) >= score_threhold:
                    bbox = f"{left} {top} {right} {bottom}"
                    bounding_boxes_per_class[tmp_class_name].append({
                        "confidence": confidence,
                        "file_id": file_id,
                        "bbox": bbox
                    })
        return bounding_boxes_per_class

    def parse_detection_line(line):
        line_split = line.split()
        tmp_class_name = " ".join(line_split[:-5])
        confidence, left, top, right, bottom = line_split[-5:]
        return tmp_class_name, confidence, left, top, right, bottom

    def save_bounding_boxes(bounding_boxes_per_class, temp_files_path):
        for class_name, bounding_boxes in bounding_boxes_per_class.items():
            bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
            with open(os.path.join(temp_files_path, f"{class_name}_dr.json"), 'w', encoding="utf-8") as outfile:
                json.dump(bounding_boxes, outfile)

    create_directories([temp_files_path, result_files_path])
    gt_counter_per_class = count_gt_classes(gt_path)
    gt_classes = sorted(gt_counter_per_class.keys())
    dr_files_list = sorted(glob.glob(dr_path + '/*.txt'))

    bounding_boxes_per_class = load_bounding_boxes(dr_files_list, gt_classes, score_threhold)
    save_bounding_boxes(bounding_boxes_per_class, temp_files_path)

    sum_ap, ap_dictionary, lamr_dictionary = 0.0, {}, {}
    for class_name in gt_classes:
        ap, lamr = compute_ap_and_lamr_for_class(class_name, minoverlap, temp_files_path, gt_counter_per_class)
        sum_ap += ap
        ap_dictionary[class_name] = ap
        lamr_dictionary[class_name] = lamr

    map_num = sum_ap / len(gt_classes)
    save_results(result_files_path, ap_dictionary, lamr_dictionary, map_num)

    if draw_plot:
        plot_results(result_files_path, gt_counter_per_class, lamr_dictionary, ap_dictionary, map_num)

    return map_num

# Additional helper functions to simplify get_map

def create_directories(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def count_gt_classes(gt_path):
    gt_counter_per_class = {}
    for txt_file in glob.glob(gt_path + '/*.txt'):
        lines_list = file_lines_to_list(txt_file)
        for line in lines_list:
            class_name = line.split()[0]
            gt_counter_per_class[class_name] = gt_counter_per_class.get(class_name, 0) + 1
    return gt_counter_per_class

def compute_ap_and_lamr_for_class(class_name, minoverlap, temp_files_path, gt_counter_per_class):
    """Load detection-results and ground-truth for the class"""
    dr_file = os.path.join(temp_files_path, f"{class_name}_dr.json")
    with open(dr_file, 'r', encoding="utf-8") as f:
        dr_data = json.load(f)

    gt_file = os.path.join(temp_files_path, f"{class_name}_ground_truth.json")
    with open(gt_file, 'r', encoding="utf-8") as f:
        gt_data = json.load(f)

    # Initialize variables
    nd = len(dr_data)
    tp = [0] * nd
    fp = [0] * nd
    score = [0] * nd
    gt_match = {key: [False] * len(gt_data[key]) for key in gt_data}
    total_gt_class = gt_counter_per_class[class_name]

    # Iterate through detection-results
    for idx, detection in enumerate(dr_data):
        file_id = detection["file_id"]
        score[idx] = float(detection["confidence"])
        bbox = [float(x) for x in detection["bbox"].split()]

        if file_id in gt_data:
            gt_bboxes = gt_data[file_id]
            iou_max = -1
            matched_gt_idx = -1

            for gt_idx, gt_bbox in enumerate(gt_bboxes):
                iou = calculate_iou(bbox, gt_bbox)
                if iou > iou_max:
                    iou_max = iou
                    matched_gt_idx = gt_idx

            if iou_max >= minoverlap and not gt_match[file_id][matched_gt_idx]:
                tp[idx] = 1
                gt_match[file_id][matched_gt_idx] = True
            else:
                fp[idx] = 1
        else:
            fp[idx] = 1

    # Compute precision-recall curve
    cumsum_fp = np.cumsum(fp)
    cumsum_tp = np.cumsum(tp)
    rec = cumsum_tp / total_gt_class
    prec = np.divide(cumsum_tp, (cumsum_fp + cumsum_tp))

    # Compute Average Precision (AP)
    ap = voc_ap(rec, prec)

    # Compute Log-Average Miss Rate (LAMR)
    fppi = cumsum_fp / total_gt_class
    lamr = log_average_miss_rate(rec, fppi, nd)

    return ap, lamr

def save_results(result_files_path, ap_dictionary, lamr_dictionary, map_num):
    with open(os.path.join(result_files_path, "results.txt"), 'w', encoding="utf-8") as results_file:
        for class_name, ap in ap_dictionary.items():
            lamr = lamr_dictionary[class_name]
            results_file.write(f"{class_name}: AP = {ap:.2f}, LAMR = {lamr:.2f}\n")
        results_file.write(f"\nMean Average Precision (mAP): {map_num:.2f}\n")

def plot_results(result_files_path, gt_counter_per_class, lamr_dictionary, ap_dictionary, map_num):
    """Plotting ground-truth info, lamr, and mAP"""
    draw_plot_func(gt_counter_per_class, len(gt_counter_per_class), "ground-truth-info",
                   "ground-truth\n", "Number of objects per class",
                   os.path.join(result_files_path, "ground-truth-info.png"),
                   False, 'forestgreen', '')
    draw_plot_func(lamr_dictionary, len(lamr_dictionary), "lamr", "log-average miss rate",
                   "log-average miss rate", os.path.join(result_files_path, "lamr.png"),
                   False, 'royalblue', "")
    draw_plot_func(ap_dictionary, len(ap_dictionary), "mAP",
                   f"mAP = {map_num * 100:.2f}%", "Average Precision",
                   os.path.join(result_files_path, "mAP.png"), False, 'royalblue', "")

def calculate_iou(bbox1, bbox2):
    """Calculate intersection over union (IoU) for two bounding boxes"""
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou
