import cv2
import numpy as np


MODEL_IMAGE_SIZE = 224
MIN_LEAF_AREA = 500
MAX_SEGMENTATION_SIDE = 1600


def _fill_holes(mask):
    height, width = mask.shape
    padded = np.pad(mask, 1, mode="constant", constant_values=0)
    flood = padded.copy()
    flood_mask = np.zeros((height + 4, width + 4), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_crop = flood[1:-1, 1:-1]
    return cv2.bitwise_or(mask, cv2.bitwise_not(flood_crop))


def segment_leaf(image_bgr):
    height, width = image_bgr.shape[:2]
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    green_gold = cv2.inRange(hsv, np.array([10, 36, 35]), np.array([95, 255, 255]))
    brown = cv2.inRange(hsv, np.array([0, 38, 35]), np.array([25, 255, 220]))
    reddish = cv2.inRange(hsv, np.array([165, 38, 35]), np.array([180, 255, 220]))
    color_mask = cv2.bitwise_or(cv2.bitwise_or(green_gold, brown), reddish)
    yellow_mask = cv2.inRange(lab[:, :, 2], 134, 255)
    leaf_core = cv2.bitwise_and(color_mask, yellow_mask)

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_core = cv2.morphologyEx(leaf_core, cv2.MORPH_OPEN, clean_kernel)
    core_contours, _ = cv2.findContours(
        leaf_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    raw_mask = color_mask.copy()

    if core_contours:
        main_core = max(core_contours, key=cv2.contourArea)
        if cv2.contourArea(main_core) > 400:
            x, y, box_width, box_height = cv2.boundingRect(main_core)
            grabcut_mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
            grabcut_mask[:4, :] = cv2.GC_BGD
            grabcut_mask[-4:, :] = cv2.GC_BGD
            grabcut_mask[:, :4] = cv2.GC_BGD
            grabcut_mask[:, -4:] = cv2.GC_BGD

            padding = 12
            x1 = max(4, x - padding)
            y1 = max(4, y - padding)
            x2 = min(width - 4, x + box_width + padding)
            y2 = min(height - 4, y + box_height + padding)
            if x1 < x2 and y1 < y2:
                grabcut_mask[y1:y2, x1:x2] = cv2.GC_PR_FGD

            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            core_eroded = cv2.morphologyEx(leaf_core, cv2.MORPH_ERODE, erode_kernel)
            grabcut_mask[core_eroded > 0] = cv2.GC_FGD
            background_model = np.zeros((1, 65), np.float64)
            foreground_model = np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(
                    image_bgr,
                    grabcut_mask,
                    None,
                    background_model,
                    foreground_model,
                    iterCount=3,
                    mode=cv2.GC_INIT_WITH_MASK,
                )
                raw_mask = np.where(
                    (grabcut_mask == cv2.GC_FGD)
                    | (grabcut_mask == cv2.GC_PR_FGD),
                    255,
                    0,
                ).astype(np.uint8)
            except cv2.error:
                raw_mask = color_mask

    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_mask = np.zeros((height, width), dtype=np.uint8)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(main_contour) > MIN_LEAF_AREA:
            cv2.drawContours(leaf_mask, [main_contour], -1, 255, thickness=cv2.FILLED)

    leaf_mask = _fill_holes(leaf_mask)
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, smooth_kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, smooth_kernel)

    if cv2.countNonZero(leaf_mask) < MIN_LEAF_AREA:
        fallback_contours, _ = cv2.findContours(
            color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        leaf_mask.fill(0)
        if fallback_contours:
            fallback_contour = max(fallback_contours, key=cv2.contourArea)
            if cv2.contourArea(fallback_contour) > MIN_LEAF_AREA:
                cv2.drawContours(
                    leaf_mask, [fallback_contour], -1, 255, thickness=cv2.FILLED
                )
                leaf_mask = _fill_holes(leaf_mask)

    dark_mask = cv2.inRange(hsv, np.array([0, 0, 10]), np.array([180, 255, 65]))
    rot_mask = cv2.inRange(hsv, np.array([0, 25, 20]), np.array([22, 255, 140]))
    defect_mask = cv2.bitwise_or(dark_mask, rot_mask)
    defect_mask = cv2.bitwise_and(defect_mask, defect_mask, mask=leaf_mask)
    defect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, defect_kernel)

    leaf_pixels = cv2.countNonZero(leaf_mask)
    defect_pixels = cv2.countNonZero(defect_mask)
    defect_percentage = (
        (defect_pixels / leaf_pixels) * 100.0 if leaf_pixels else 0.0
    )
    return leaf_mask, round(float(defect_percentage), 1)


def prepare_leaf_for_model(image_bgr, leaf_mask):
    contours, _ = cv2.findContours(
        leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("Daun tidak terdeteksi; gunakan foto daun dengan kontras jelas.")

    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) <= MIN_LEAF_AREA:
        raise ValueError("Area daun terlalu kecil untuk diklasifikasikan.")

    component_mask = np.zeros(leaf_mask.shape, dtype=np.uint8)
    cv2.drawContours(component_mask, [main_contour], -1, 255, thickness=cv2.FILLED)
    x, y, width, height = cv2.boundingRect(main_contour)
    isolated = cv2.bitwise_and(image_bgr, image_bgr, mask=component_mask)
    crop = isolated[y:y + height, x:x + width]

    side = max(width, height)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    offset_x = (side - width) // 2
    offset_y = (side - height) // 2
    square[offset_y:offset_y + height, offset_x:offset_x + width] = crop
    resized = cv2.resize(
        square,
        (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)


def prepare_leaf_sample(image_bgr):
    height, width = image_bgr.shape[:2]
    scale = min(1.0, MAX_SEGMENTATION_SIDE / max(height, width))
    if scale < 1.0:
        working_image = cv2.resize(
            image_bgr,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        working_image = image_bgr

    leaf_mask, defect_percentage = segment_leaf(working_image)
    if cv2.countNonZero(leaf_mask) < MIN_LEAF_AREA:
        raise ValueError(
            "Daun tidak berhasil dipisahkan dari latar belakang. "
            "Gunakan foto dengan daun utuh dan kontras yang jelas."
        )
    model_image = prepare_leaf_for_model(working_image, leaf_mask)
    return working_image, leaf_mask, defect_percentage, model_image


def make_transparent_leaf_preview(image_bgr, leaf_mask):
    contours, _ = cv2.findContours(
        leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("Daun tidak terdeteksi; hasil segmentasi tidak dapat dibuat.")

    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) <= MIN_LEAF_AREA:
        raise ValueError("Area daun terlalu kecil untuk dibuatkan hasil segmentasi.")

    component_mask = np.zeros(leaf_mask.shape, dtype=np.uint8)
    cv2.drawContours(component_mask, [main_contour], -1, 255, thickness=cv2.FILLED)
    alpha = cv2.GaussianBlur(component_mask, (3, 3), 0.6)
    preview = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    preview[:, :, 3] = alpha
    preview[alpha == 0, :3] = 0
    return preview
