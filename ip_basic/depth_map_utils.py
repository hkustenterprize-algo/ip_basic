import collections

import cv2
import numpy as np

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=np.uint8,
)

# 3 x 3 cross kernel
CROSS2_KERNEL_3 = np.asarray(
    [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ],
    dtype=np.uint8,
)

STAR_KERNEL_3 = np.asarray(
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.uint8,
)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ],
    dtype=np.uint8,
)

# 5x5 cross kernel
CROSS2_KERNEL_5 = np.asarray(
    [
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1],
    ],
    dtype=np.uint8,
)

STAR_KERNEL_5 = np.asarray(
    [
        [1, 0, 1, 0, 1],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [1, 0, 1, 0, 1],
    ],
    dtype=np.uint8,
)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ],
    dtype=np.uint8,
)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ],
    dtype=np.uint8,
)

# 7x7 cross kernel
CROSS2_KERNEL_7 = np.asarray(
    [
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
    ],
    dtype=np.uint8,
)

STAR_KERNEL_7 = np.asarray(
    [
        [1, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1],
    ],
    dtype=np.uint8,
)
# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ],
    dtype=np.uint8,
)

STAR_KERNEL_9 = np.zeros((9, 9), dtype=np.uint8)
STAR_KERNEL_9[4, :] = 1
STAR_KERNEL_9[:, 4] = 1
STAR_KERNEL_9[[1, 2, 3, 5, 6, 7], [1, 2, 3, 5, 6, 7]] = 1
STAR_KERNEL_9[[1, 2, 3, 5, 6, 7], [7, 6, 5, 3, 2, 1]] = 1

# 11x11 星型核
STAR_KERNEL_11 = np.zeros((11, 11), dtype=np.uint8)
STAR_KERNEL_11[5, :] = 1
STAR_KERNEL_11[:, 5] = 1
STAR_KERNEL_11[[1, 2, 3, 4, 6, 7, 8, 9], [1, 2, 3, 4, 6, 7, 8, 9]] = 1
STAR_KERNEL_11[[1, 2, 3, 4, 6, 7, 8, 9], [9, 8, 7, 6, 4, 3, 2, 1]] = 1

# 15x15 星型核
STAR_KERNEL_15 = np.zeros((15, 15), dtype=np.uint8)
STAR_KERNEL_15[7, :] = 1
STAR_KERNEL_15[:, 7] = 1
STAR_KERNEL_15[
    [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13], [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
] = 1
STAR_KERNEL_15[
    [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13], [13, 12, 11, 10, 9, 8, 6, 5, 4, 3, 2, 1]
] = 1


def compute_distance_map(height, width):
    """计算每个像素到中心的归一化距离"""
    center_y, center_x = height / 2, width / 2  # 中心点 (259, 259)
    y, x = np.indices((height, width))
    distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_distance = np.sqrt(center_x**2 + center_y**2)  # 最大距离（中心到角落）
    distance_map = distances / max_distance  # 归一化到 [0, 1]
    return distance_map


DISTANCE_MAP = compute_distance_map(518, 518)


def get_kernel_by_distance(distance):
    """根据距离选择星型核，中心小核，边缘大核"""
    if distance < 0.3:
        return CROSS_KERNEL_3  # 中心
    elif distance < 0.6:
        return CROSS_KERNEL_5  # 中间
    else:
        return CROSS_KERNEL_7  # 边缘


def adaptive_filter(depth, valid_mask, distance_map):
    """修复版自适应滤波，基于空间距离选择核大小，确保深度传播"""
    filtered = np.zeros_like(depth)

    working_depth = np.zeros_like(depth)
    working_depth[valid_mask] = depth[valid_mask]

    distance_bins = [0.3, 0.6]
    masks = [
        distance_map < distance_bins[0],  # 中心
        (distance_map >= distance_bins[0]) & (distance_map < distance_bins[1]),  # 中间
        distance_map >= distance_bins[1],  # 边缘
    ]

    for mask, bin_distance in zip(masks, [0.1, 0.45, 0.8]):
        kernel = get_kernel_by_distance(bin_distance)
        filtered_region = cv2.dilate(working_depth, kernel)
        working_depth = np.where(mask, filtered_region, working_depth)

    filtered = working_depth
    filtered[valid_mask] = depth[valid_mask]
    return filtered


def fill_in_fast(
    depth_map,
    max_depth=100.0,
    custom_kernel=DIAMOND_KERNEL_5,
    extrapolate=False,
    blur_type="bilateral",
):
    """Fast, in-place depth completion.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_map: dense depth map
    """

    # Invert
    valid_pixels = depth_map > 0.1
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = depth_map < 0.1
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0 : top_row_pixels[pixel_col_idx], pixel_col_idx] = (
                top_pixel_values[pixel_col_idx]
            )

        # Large Fill
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = cv2.medianBlur(depth_map, 5)

    # Bilateral or Gaussian blur
    if blur_type == "bilateral":
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == "gaussian":
        # Gaussian blur
        valid_pixels = depth_map > 0.1
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = depth_map > 0.1
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map


def fill_in_multiscale(
    depth_map,
    max_depth=100.0,
    dilation_kernel_far=CROSS_KERNEL_3,
    dilation_kernel_med=CROSS_KERNEL_5,
    dilation_kernel_near=CROSS_KERNEL_7,
    extrapolate=False,
    blur_type="bilateral",
    show_process=True,
):
    """Slower, multi-scale dilation version with additional noise removal that
    provides better qualitative results.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        dilation_kernel_far: dilation kernel to use for 30.0 < depths < 80.0 m
        dilation_kernel_med: dilation kernel to use for 15.0 < depths < 30.0 m
        dilation_kernel_near: dilation kernel to use for 0.1 < depths < 15.0 m
        extrapolate:whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'gaussian' - provides lower RMSE
            'bilateral' - preserves local structure (recommended)
        show_process: saves process images into an OrderedDict

    Returns:
        depth_map: dense depth map
        process_dict: OrderedDict of process images
    """

    # Convert to float32

    depths_in = np.float32(depth_map)

    # Calculate bin masks before inversion
    valid_pixels_near = (depths_in > 0.1) & (depths_in <= 15.0)
    valid_pixels_med = (depths_in > 15.0) & (depths_in <= 30.0)
    valid_pixels_far = depths_in > 30.0

    # Invert (and offset)
    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = s1_inverted_depths > 0.1
    s1_inverted_depths[valid_pixels] = max_depth - s1_inverted_depths[valid_pixels]

    # Multi-scale dilation
    # dilated_far = cv2.dilate(
    #     np.multiply(s1_inverted_depths, valid_pixels_far), dilation_kernel_far
    # )
    # dilated_med = cv2.dilate(
    #     np.multiply(s1_inverted_depths, valid_pixels_med), dilation_kernel_med
    # )
    # dilated_near = cv2.dilate(
    #     np.multiply(s1_inverted_depths, valid_pixels_near), dilation_kernel_near
    # )

    dilated_far = adaptive_filter(
        s1_inverted_depths,
        valid_pixels_far,
        DISTANCE_MAP,
    )
    dilated_med = adaptive_filter(
        s1_inverted_depths,
        valid_pixels_med,
        DISTANCE_MAP,
    )
    dilated_near = adaptive_filter(
        s1_inverted_depths,
        valid_pixels_near,
        DISTANCE_MAP,
    )

    # Find valid pixels for each binned dilation
    valid_pixels_near = dilated_near > 0.1
    valid_pixels_med = dilated_med > 0.1
    valid_pixels_far = dilated_far > 0.1

    # Combine dilated versions, starting farthest to nearest
    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    # Small hole closure
    s3_closed_depths = cv2.morphologyEx(
        s2_dilated_depths, cv2.MORPH_CLOSE, FULL_KERNEL_5
    )

    # Median blur to remove outliers
    s4_blurred_depths = np.copy(s3_closed_depths)
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = s3_closed_depths > 0.1
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Calculate a top mask
    top_mask = np.ones(depths_in.shape, dtype=bool)
    for pixel_col_idx in range(s4_blurred_depths.shape[1]):
        pixel_col = s4_blurred_depths[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.1)
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    # Get empty mask
    valid_pixels = s4_blurred_depths > 0.1
    empty_pixels = ~valid_pixels & top_mask

    # Hole fill
    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_9)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image or create top mask
    s6_extended_depths = np.copy(s5_dilated_depths)
    # top_mask = np.ones(s5_dilated_depths.shape, dtype=bool)

    # top_row_pixels = np.argmax(s5_dilated_depths > 0.1, axis=0)
    # top_pixel_values = s5_dilated_depths[
    #     top_row_pixels, range(s5_dilated_depths.shape[1])
    # ]

    # for pixel_col_idx in range(s5_dilated_depths.shape[1]):
    #     if extrapolate:
    #         s6_extended_depths[0 : top_row_pixels[pixel_col_idx], pixel_col_idx] = (
    #             top_pixel_values[pixel_col_idx]
    #         )
    #     else:
    #         # Create top mask
    #         top_mask[0 : top_row_pixels[pixel_col_idx], pixel_col_idx] = False

    # Fill large holes with masked dilations
    s7_blurred_depths = np.copy(s6_extended_depths)
    for i in range(6):
        empty_pixels = s7_blurred_depths < 0.1
        dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_7)
        s7_blurred_depths[empty_pixels] = dilated[empty_pixels]
        blurred = cv2.medianBlur(s7_blurred_depths, 5)
        valid_pixels = s7_blurred_depths > 0.1
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Median blur
    # blurred = cv2.medianBlur(s7_blurred_depths, 5)
    # valid_pixels = (s7_blurred_depths > 0.1) & top_mask
    # s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    if blur_type == "gaussian":
        # Gaussian blur
        blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)
        valid_pixels = (s7_blurred_depths > 0.1) & top_mask
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    elif blur_type == "bilateral":
        # Bilateral blur
        blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 100)
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Invert (and offset)
    s8_inverted_depths = np.copy(s7_blurred_depths)
    valid_pixels = np.where(s8_inverted_depths > 0.1)
    s8_inverted_depths[valid_pixels] = max_depth - s8_inverted_depths[valid_pixels]

    depths_out = s8_inverted_depths

    process_dict = None
    if show_process:
        process_dict = collections.OrderedDict()

        process_dict["s0_depths_in"] = depths_in

        process_dict["s1_inverted_depths"] = s1_inverted_depths
        process_dict["s2_dilated_depths"] = s2_dilated_depths
        process_dict["s3_closed_depths"] = s3_closed_depths
        process_dict["s4_blurred_depths"] = s4_blurred_depths
        process_dict["s5_combined_depths"] = s5_dilated_depths
        process_dict["s6_extended_depths"] = s6_extended_depths
        process_dict["s7_blurred_depths"] = s7_blurred_depths
        process_dict["s8_inverted_depths"] = s8_inverted_depths

        process_dict["s9_depths_out"] = depths_out

    for key, val in process_dict.items():
        print(key, "is nan{}".format(np.isnan(val).astype(np.uint8).sum()))

    return depths_out, process_dict
