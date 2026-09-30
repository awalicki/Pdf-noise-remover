import os
import datetime
from typing import List, Optional
import numpy as np
import cv2
from PIL import Image

# pip install numpy opencv-python Pillow pymupdf

# Source - https://stackoverflow.com/a/76297437
# Posted by questionto42
# Retrieved 2026-09-30, License - CC BY-SA 4.0

# echo $Env:VIRTUAL_ENV


OUTPUT_DIR = rf"Your\dir"

USE_CONNECTED_COMPONENTS = True
MIN_HEIGHT = 18
MIN_WIDTH = 9
MIN_AREA = 0

USE_MORPH_OPEN = False
MORPH_OPEN_KERNEL_SIZE = 2

USE_MORPH_CLOSE = True
MORPH_CLOSE_KERNEL_SIZE = 2

THRESHOLD_METHOD = "otsu"
ADAPTIVE_THRESH_BLOCK_SIZE = 81
ADAPTIVE_THRESH_C = 12
WHITEN_THRESHOLD = 255

PRESERVE_ORIGINAL_COLOR = True
DPI = 200
BYPASS_CLEANING = False


def load_pdf_pages_as_images(pdf_path: str, dpi: int = DPI) -> List[np.ndarray]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Invalid input path: {pdf_path}")

    try:
        import fitz
        doc = fitz.open(pdf_path)
        images = []
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 3:
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            elif pix.n == 1:
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img_data
            images.append(img_bgr)

        return images
    except ImportError:
        raise RuntimeError("brak pymupdf")


def clean_page_image(
        image: np.ndarray,
        min_height: int = MIN_HEIGHT,
        min_width: int = MIN_WIDTH,
        min_area: int = MIN_AREA,
        use_morph_open: bool = USE_MORPH_OPEN,
        morph_open_kernel: int = MORPH_OPEN_KERNEL_SIZE,
        use_morph_close: bool = USE_MORPH_CLOSE,
        morph_close_kernel: int = MORPH_CLOSE_KERNEL_SIZE,
        use_connected_components: bool = USE_CONNECTED_COMPONENTS,
        threshold_method: str = THRESHOLD_METHOD,
        adaptive_block_size: int = ADAPTIVE_THRESH_BLOCK_SIZE,
        adaptive_c: int = ADAPTIVE_THRESH_C,
        preserve_original_color: bool = PRESERVE_ORIGINAL_COLOR,
        whiten_threshold: int = WHITEN_THRESHOLD,
        bypass_cleaning: bool = BYPASS_CLEANING,
) -> np.ndarray:
    if bypass_cleaning:
        return image.copy()

    image_copy = image.copy()

    if len(image_copy.shape) == 3:
        gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_copy.copy()

    if whiten_threshold < 255:
        white_mask = gray > whiten_threshold
        gray[white_mask] = 255
        if len(image_copy.shape) == 3:
            image_copy[white_mask] = [255, 255, 255]

    if threshold_method.lower() == "adaptive":
        block_size = adaptive_block_size if adaptive_block_size % 2 == 1 else adaptive_block_size + 1
        binary_inv = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, adaptive_c
        )
    else:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    working_mask = binary_inv.copy()

    if use_morph_open and morph_open_kernel > 0:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_open_kernel, morph_open_kernel))
        working_mask = cv2.morphologyEx(working_mask, cv2.MORPH_OPEN, kernel_open)

    if use_morph_close and morph_close_kernel > 0:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_close_kernel, morph_close_kernel))
        working_mask = cv2.morphologyEx(working_mask, cv2.MORPH_CLOSE, kernel_close)

    if use_connected_components:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            working_mask, connectivity=8
        )

        widths = stats[:, cv2.CC_STAT_WIDTH]
        heights = stats[:, cv2.CC_STAT_HEIGHT]
        areas = stats[:, cv2.CC_STAT_AREA]

        valid_labels_mask = (heights >= min_height) & (widths >= min_width) & (areas >= min_area)
        valid_labels_mask[0] = False

        lut = np.zeros(num_labels, dtype=np.uint8)
        lut[valid_labels_mask] = 255

        working_mask = lut[labels]

    if preserve_original_color and len(image_copy.shape) == 3:
        output_image = np.full_like(image_copy, 255)
        text_pixels = working_mask > 0
        output_image[text_pixels] = image_copy[text_pixels]
    else:
        output_gray = cv2.bitwise_not(working_mask)
        output_image = cv2.cvtColor(output_gray, cv2.COLOR_GRAY2BGR)

    return output_image


def process_document(
        input_path: str,
        output_path: Optional[str] = None,
        min_height: int = MIN_HEIGHT,
        min_width: int = MIN_WIDTH,
        min_area: int = MIN_AREA,
        use_morph_open: bool = USE_MORPH_OPEN,
        morph_open_kernel: int = MORPH_OPEN_KERNEL_SIZE,
        use_morph_close: bool = USE_MORPH_CLOSE,
        morph_close_kernel: int = MORPH_CLOSE_KERNEL_SIZE,
        use_connected_components: bool = USE_CONNECTED_COMPONENTS,
        threshold_method: str = THRESHOLD_METHOD,
        adaptive_block_size: int = ADAPTIVE_THRESH_BLOCK_SIZE,
        adaptive_c: int = ADAPTIVE_THRESH_C,
        preserve_original_color: bool = PRESERVE_ORIGINAL_COLOR,
        dpi: int = DPI,
        whiten_threshold: int = WHITEN_THRESHOLD,
        bypass_cleaning: bool = BYPASS_CLEANING,
) -> str:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Invalid input file: {input_path}")

    if not output_path:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_{min_height}_cleaned.pdf"

    is_pdf = input_path.lower().endswith(".pdf")

    if is_pdf:
        pages = load_pdf_pages_as_images(input_path, dpi=dpi)
    else:
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"file is unreadable: {input_path}")
        pages = [img]

    cleaned_pil_pages = []

    for idx, page in enumerate(pages, start=1):
        cleaned_bgr = clean_page_image(
            image=page,
            min_height=min_height,
            min_width=min_width,
            min_area=min_area,
            use_morph_open=use_morph_open,
            morph_open_kernel=morph_open_kernel,
            use_morph_close=use_morph_close,
            morph_close_kernel=morph_close_kernel,
            use_connected_components=use_connected_components,
            threshold_method=threshold_method,
            adaptive_block_size=adaptive_block_size,
            adaptive_c=adaptive_c,
            preserve_original_color=preserve_original_color,
            whiten_threshold=whiten_threshold,
            bypass_cleaning=bypass_cleaning,
        )
        cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cleaned_rgb)
        cleaned_pil_pages.append(pil_img)

    if cleaned_pil_pages:
        cleaned_pil_pages[0].save(
            output_path,
            save_all=True,
            append_images=cleaned_pil_pages[1:],
            resolution=float(dpi),
        )
        return output_path

    raise RuntimeError("There's nothing to save.")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i in range(3, 6):
        out = process_document(
            input_path=rf"Your output path, I used loop to run script one and process multiple files",
            output_path=os.path.join(OUTPUT_DIR,
                                     f"{i}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_cleaned.pdf"),
        )