import numpy as np
import matplotlib.pyplot as plt


def show_mask(mask, ax, obj_id=None, random_color=False, transparency=0.6):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([1 - transparency])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 1 - transparency])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# given a DLC labels, extract the coordinate pairs for a given row index, skipping NaN values
def get_coordinates(df, row_idx):
    """Extract coordinate pairs from a row, skipping NaN values."""
    row = df.iloc[row_idx]
    coords = []
    # get all column names ending with _x, find matching _y
    x_cols = [c for c in df.columns if c.endswith('_x')]
    for x_col in x_cols:
        y_col = x_col[:-2] + '_y'
        if y_col in df.columns:
            x, y = row[x_col], row[y_col]
            if not (np.isnan(x) or np.isnan(y)):
                coords.append([int(x), int(y)])
    return coords
