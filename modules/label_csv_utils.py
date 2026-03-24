import csv
import re

import pandas as pd


def _detect_newline(csv_path):
    with open(csv_path, "rb") as f:
        content = f.read()
    if b"\r\n" in content:
        return "\r\n"
    return "\n"


def _infer_frame_padding_width(frame_values):
    widths = []
    for value in frame_values:
        if not isinstance(value, str):
            continue
        match = re.match(r"img(\d+)\.png$", value.strip())
        if match:
            widths.append(len(match.group(1)))
    if widths:
        return max(widths)
    return None


def _format_dlc_frame_name(frame_idx, width):
    frame_idx = int(frame_idx)
    return f"img{str(frame_idx).zfill(width)}.png"


def _serialize_keypoint_value(value):
    if pd.isna(value):
        return ""
    return str(value)


def load_keypoints(csv_path):
    """
    Load a DLC CSV and return a DataFrame with:
      - first column: numeric frame index
      - remaining columns: keypoint x/y columns in the same order as the CSV

    Expected DLC CSV layout:
      row 0: scorer
      row 1: bodyparts
      row 2: coords
      row 3+: data rows, with frame/image name in column 2
    """
    with open(csv_path, "r", newline="") as f:
        raw_rows = list(csv.reader(f))

    df = pd.read_csv(csv_path, header=[0, 1, 2])
    if df.shape[1] < 4:
        return pd.DataFrame(columns=["frame"])

    frame_names = df.iloc[:, 2].astype(str).str.strip()
    frame_names = frame_names.where(frame_names != "nan")
    frame_index = (
        frame_names.str.extract(r"(\d+)", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    out = pd.DataFrame({"frame": frame_index})

    for col in df.columns[3:]:
        if len(col) < 3:
            continue
        bodypart = str(col[1]).strip()
        coord = str(col[2]).strip().lower()
        if coord not in ("x", "y"):
            continue
        out[f"{bodypart}_{coord}"] = pd.to_numeric(df[col], errors="coerce")

    keypoint_columns = list(out.columns[1:])
    out = out.dropna(subset=["frame"]).reset_index(drop=True)
    out["frame"] = out["frame"].astype(int)

    out.attrs["dlc_header_rows"] = raw_rows[:3]
    out.attrs["dlc_first_col_value"] = raw_rows[3][0] if len(raw_rows) > 3 and raw_rows[3] else "labeled-data"
    out.attrs["dlc_frame_padding_width"] = _infer_frame_padding_width(frame_names.tolist())
    out.attrs["dlc_keypoint_columns"] = keypoint_columns
    out.attrs["dlc_newline"] = _detect_newline(csv_path)
    return out


def save_keypoints(df, video_name, csv_path):
    """
    Save a keypoint DataFrame in DLC CSV format.

    The input DataFrame is expected to have:
      - first column: frame index
      - remaining columns: bodypart_x/bodypart_y columns in DLC CSV order
    """
    if df.empty:
        raise ValueError("df is empty")

    frame_col = df.columns[0]
    keypoint_columns = list(df.columns[1:])

    header_rows = df.attrs.get("dlc_header_rows")
    if not header_rows or len(header_rows) != 3:
        raise ValueError("df is missing DLC header metadata; load it with load_keypoints first")

    expected_columns = df.attrs.get("dlc_keypoint_columns", keypoint_columns)
    if keypoint_columns != expected_columns:
        raise ValueError("df keypoint columns do not match the original DLC CSV column order")

    frame_width = df.attrs.get("dlc_frame_padding_width")
    if frame_width is None:
        max_frame = int(pd.to_numeric(df[frame_col], errors="coerce").max())
        frame_width = max(1, len(str(max_frame)))

    first_col_value = df.attrs.get("dlc_first_col_value", "labeled-data")
    newline = df.attrs.get("dlc_newline", "\n")

    rows = [list(row) for row in header_rows]

    for _, row in df.iterrows():
        frame_idx = pd.to_numeric(row[frame_col], errors="coerce")
        if pd.isna(frame_idx):
            continue
        frame_idx = int(frame_idx)

        data_row = [
            first_col_value,
            video_name,
            _format_dlc_frame_name(frame_idx, frame_width),
        ]

        for col_name in keypoint_columns:
            data_row.append(_serialize_keypoint_value(row[col_name]))

        rows.append(data_row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator=newline)
        writer.writerows(rows)


def frame_idx_to_image_name(frame_idx, padding_width):
    return _format_dlc_frame_name(frame_idx, padding_width)


def get_image_names(df):
    padding_width = df.attrs.get("dlc_frame_padding_width", 1)
    frame_col = df.columns[0]
    return [
        _format_dlc_frame_name(frame_idx, padding_width)
        for frame_idx in pd.to_numeric(df[frame_col], errors="coerce").dropna().astype(int)
    ]


def round_keypoints(df):
    """
    Return a copy of the keypoint DataFrame with frame/x/y columns rounded
    to the nearest integer.
    """
    rounded = df.copy(deep=True)
    for col in rounded.columns:
        try:
            rounded[col] = pd.to_numeric(rounded[col])
        except (TypeError, ValueError):
            pass

    numeric_cols = [
        col for col in rounded.columns
        if col == "frame" or col.endswith("_x") or col.endswith("_y")
    ]

    for col in numeric_cols:
        rounded[col] = rounded[col].round().astype("Int64")

    return rounded
