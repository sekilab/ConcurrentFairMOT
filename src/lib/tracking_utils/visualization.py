import numpy as np
import cv2


def tlwhs_to_tlbrs(tlwhs):
    tlbrs = np.copy(tlwhs)
    if len(tlbrs) == 0:
        return tlbrs
    tlbrs[:, 2] += tlwhs[:, 0]
    tlbrs[:, 3] += tlwhs[:, 1]
    return tlbrs


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def resize_image(image, max_size=800):
    if max(image.shape[:2]) > max_size:
        scale = float(max_size) / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale)
    return image


def plot_tracking(images, tlwhs, obj_ids, prev_tlwhs=None, prev_ids=None, scores=None, frame_id=0, fps=0., ids2=None):
    if len(images) < 1:
        raise Exception(f'images array has invalid length of {len(images)}')
    if len(images) > 1 and (prev_tlwhs is None or prev_ids is None):
        raise Exception(f'image series passed, but no previous processed frame information available')

    ret_ims = []
    for i, image in enumerate(images):
        factor = (i+1) / len(images)
        im = np.ascontiguousarray(np.copy(image))
        im_h, im_w = im.shape[:2]

        top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

        text_scale = max(1, image.shape[1] / 1600.)
        text_thickness = 2
        line_thickness = max(1, int(image.shape[1] / 500.))

        radius = max(5, int(im_w/140.))
        #cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
        #            (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)
        cv2.putText(im, 'frame: %d num: %d' % ((frame_id - len(images) + i + 1), len(tlwhs)),
                    (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

        for j, tlwh in enumerate(tlwhs):
            obj_id = obj_ids[j]
            x1, y1, w, h = tlwh
            if i < len(images) - 1:
                prev_index = prev_ids.index(obj_id) if obj_id in prev_ids else None
                if prev_index is None:
                    continue
                px1, py1, pw, ph = prev_tlwhs[prev_index]
                x1 = px1 + (x1 - px1) * factor
                y1 = py1 + (y1 - py1) * factor
                w = pw + (w - pw) * factor
                h = ph + (h - ph) * factor
            obj_id = int(obj_id)
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            #id_text = '{}'.format(int(obj_id))
            #if ids2 is not None:
            #    id_text = id_text + ', {}'.format(int(ids2[j]))
            _line_thickness = 1 if obj_id <= 0 else line_thickness
            color = get_color(abs(obj_id))
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
            #cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
            #            thickness=text_thickness)
        ret_ims.append(im)
    return ret_ims


def plot_trajectory(image, tlwhs, track_ids):
    image = image.copy()
    for one_tlwhs, track_id in zip(tlwhs, track_ids):
        color = get_color(int(track_id))
        for tlwh in one_tlwhs:
            x1, y1, w, h = tuple(map(int, tlwh))
            cv2.circle(image, (int(x1 + 0.5 * w), int(y1 + h)), 2, color, thickness=2)

    return image


def plot_detections(image, tlbrs, scores=None, color=(255, 0, 0), ids=None):
    im = np.copy(image)
    text_scale = max(1, image.shape[1] / 800.)
    thickness = 2 if text_scale > 1.3 else 1
    for i, det in enumerate(tlbrs):
        x1, y1, x2, y2 = np.asarray(det[:4], dtype=np.int)
        if len(det) >= 7:
            label = 'det' if det[5] > 0 else 'trk'
            if ids is not None:
                text = '{}# {:.2f}: {:d}'.format(label, det[6], ids[i])
                cv2.putText(im, text, (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                            thickness=thickness)
            else:
                text = '{}# {:.2f}'.format(label, det[6])

        if scores is not None:
            text = '{:.2f}'.format(scores[i])
            cv2.putText(im, text, (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                        thickness=thickness)

        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)

    return im
