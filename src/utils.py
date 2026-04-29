import numpy as np
import pydicom as dicom
import zipfile
import os
import io
from scipy.ndimage import zoom


def build_image_index(
    image_folder=None,
    zip_path=None,
    load_from_zip=False
):
    """
    Builds image_index and returns (image_index, zip_file)

    image_index:
        dict {image_id: path_or_zip_internal_path}

    zip_file:
        zipfile.ZipFile object or None
    """

    image_index = {}
    z = None

    if load_from_zip:
        if zip_path is None:
            raise ValueError("zip_path must be provided when load_from_zip=True")

        z = zipfile.ZipFile(zip_path, 'r')

        for name in z.namelist():
            if name.endswith(".dicom"):
                image_id = os.path.basename(name).replace(".dicom", "")
                image_index[image_id] = name

    else:
        if image_folder is None:
            raise ValueError("image_folder must be provided when load_from_zip=False")

        for root, _, files in os.walk(image_folder):
            for file in files:
                if file.endswith(".dicom"):
                    image_id = file.replace(".dicom", "")
                    image_index[image_id] = os.path.join(root, file)

    return image_index, z

def load_image(
    image_id,
    image_index,
    load_from_zip=False,
    zip_file=None,
    normalize=True,
    fix_inversion=True
):
    """
    Universal image loader.

    Parameters:
    - image_id: str
    - image_index: dict {image_id: path or zip_path}
    - load_from_zip: bool
    - zip_file: zipfile.ZipFile object (required if load_from_zip=True)
    - normalize: bool → scale to [0,1]
    - fix_inversion: bool → handle MONOCHROME1

    Returns:
    - img (float32 numpy array) or None
    """

    if image_id not in image_index:
        return None

    try:
        # -------------------
        # LOAD DICOM
        # -------------------
        if load_from_zip:
            if zip_file is None:
                raise ValueError("zip_file must be provided when load_from_zip=True")

            with zip_file.open(image_index[image_id]) as f:
                ds = dicom.dcmread(io.BytesIO(f.read()))
        else:
            path = image_index[image_id]
            if not os.path.exists(path):
                return None
            ds = dicom.dcmread(path)

        img = ds.pixel_array.astype(np.float32)

        # -------------------
        # NORMALIZE
        # -------------------
        if normalize:
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        # -------------------
        # FIX INVERSION
        # -------------------
        if fix_inversion:
            if hasattr(ds, "PhotometricInterpretation"):
                if ds.PhotometricInterpretation == "MONOCHROME1":
                    img = 1.0 - img
            else:
                # fallback heuristic
                if np.mean(img) > 0.6:
                    img = 1.0 - img

        return img

    except Exception as e:
        print(f"[load_image ERROR] {image_id}: {e}")
        return None


def load_data(
    df,
    data_root,
    multi_scale=False,
    add_channel_dim=True
):
    """
    Load dataset into memory.

    Returns:
        if multi_scale:
            ([X_small, X_large], Y)
        else:
            (X, Y)
    """

    Y = df["label"].values.astype(np.int32)

    # -------------------
    # MULTI-SCALE
    # -------------------
    if multi_scale:
        small_dir = os.path.join(data_root, "images_small")
        large_dir = os.path.join(data_root, "images_large")

        X_small = []
        X_large = []

        for _, row in df.iterrows():
            img_small = np.load(os.path.join(small_dir, row["file_small"]))
            img_large = np.load(os.path.join(large_dir, row["file_large"]))

            X_small.append(img_small)
            X_large.append(img_large)

        X_small = np.array(X_small)
        X_large = np.array(X_large)

        if add_channel_dim:
            if X_small.ndim == 3:
                X_small = X_small[..., np.newaxis]
                X_large = X_large[..., np.newaxis]

        return [X_small, X_large], Y

    # -------------------
    # SINGLE-SCALE
    # -------------------
    else:
        img_dir = os.path.join(data_root, "images")

        X = []

        for _, row in df.iterrows():
            img = np.load(os.path.join(img_dir, row["file"]))
            X.append(img)

        X = np.array(X)

        if add_channel_dim and X.ndim == 3:
            X = X[..., np.newaxis]

        return X, Y
    
def extract_multiscale_patch(
    img,
    x,
    y,
    patch_size=64,
    scale_factor=2,
    pad_mode="constant"
):
    """
    Extracts:
    - small patch (patch_size x patch_size)
    - larger context patch (scaled by scale_factor, then resized back)

    Returns:
        p_small, p_large  (both shape: patch_size x patch_size)
    """

    # -------------------
    # SMALL PATCH
    # -------------------
    p_small = img[y:y+patch_size, x:x+patch_size]

    # Safety check (skip if out of bounds)
    if p_small.shape != (patch_size, patch_size):
        return None, None

    # -------------------
    # LARGE PATCH
    # -------------------
    large_size = patch_size * scale_factor

    y_large = max(0, y - patch_size // 2)
    x_large = max(0, x - patch_size // 2)

    p_large = img[y_large:y_large+large_size, x_large:x_large+large_size]

    # -------------------
    # PAD IF NEEDED
    # -------------------
    if p_large.shape != (large_size, large_size):
        pad_y = large_size - p_large.shape[0]
        pad_x = large_size - p_large.shape[1]

        p_large = np.pad(
            p_large,
            ((0, pad_y), (0, pad_x)),
            mode=pad_mode
        )

    # -------------------
    # RESIZE BACK
    # -------------------
    p_large = zoom(
        p_large,
        (patch_size / large_size, patch_size / large_size),
        order=1
    )

    return p_small, p_large

