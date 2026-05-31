import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


def rgb_to_class_index(mask_rgb, color_map):
    mask_idx = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
    for rgb, class_idx in color_map.items():
        matches = np.all(mask_rgb == rgb, axis=-1)
        mask_idx[matches] = class_idx

    return mask_idx


# --- Unified Class Indices ---
# 0: Background/Field/Ground
# 1: Player (Team 1, 2, A, B)
# 2: Goalkeeper (GK 1, 2, A, B)
# 3: Referee
# 4: Ball / Football
# 5: Goal Bar
# 6: Advertisement
# 7: Audience / Spectators
# 8: Staff / Coaches & Officials
CLASS_NAMES = [
        "0: Background", "1: Player", "2: Goalkeeper",
        "3: Referee", "4: Ball", "5: Goal Bar",
        "6: Advertisement", "7: Audience"
    ]

DATASET1_COLOR_MAP = {
    # (0, 0, 0): 0,  # Background -> Unified: Background (0)
    (201, 158, 74): 0,  # Field -> Unified: Background (0)
    (38, 198, 129): 1,  # Team 1 -> Unified: Player (1)
    (27, 154, 218): 1,  # Team 2 -> Unified: Player (1)
    (153, 223, 219): 2,  # Goalkeeper 1 -> Unified: Goalkeeper (2)
    (255, 106, 77): 2,  # Goalkeeper 2 -> Unified: Goalkeeper (2)
    (22, 100, 252): 3,  # Referee -> Unified: Referee (3)
    (96, 32, 192): 4,  # Football -> Unified: Ball (4)
    (89, 134, 179): 5,  # Goal Bar -> Unified: Goal Bar (5)
    (237, 34, 236): 6,  # Advertisement -> Unified: Advertisement (6)
    (143, 182, 45): 7  # Spectators -> Unified: Audience (7)
}

DATASET2_COLOR_MAP = {
    (137, 126, 126): 0,  # Ground (Background)
    (254, 233, 3): 1,  # Team A (Player)
    (255, 160, 1): 1,  # # Team B (Player)
    (255, 159, 0): 2,  # Keeper B
    (255, 235, 0): 2,  # keeper A
    (238, 171, 171): 3,  # Referee
    (201, 19, 223): 4,  # Ball
    (255, 0, 29): 5,  # Goal Bar
    (27, 71, 151): 6,  # Advertisement
    (111, 48, 253): 7,  # Audience
    # (0, 0, 0): 8  # Coaches & Officials (Staff) no samples
}


# incorrect rgb colors in doc

# DATASET2_COLOR_MAP = {
#     (0, 0, 0): 0,  # Ground (Background)
#     (27, 71, 151): 1,  # Team A (Player)
#     (111, 48, 253): 1,  # Team B (Player)
#     (137, 126, 126): 2,  # Goalkeeper A
#     (201, 19, 223): 2,  # Goalkeeper B
#     (238, 171, 171): 3,  # Referee
#     (254, 233, 3): 4,  # Ball
#     (255, 0, 29): 5,  # Goal Bar
#     (255, 159, 0): 6,  # Advertisement
#     (255, 160, 1): 7,  # Audience
#     (255, 235, 0): 8  # Coaches & Officials (Staff)
# }
