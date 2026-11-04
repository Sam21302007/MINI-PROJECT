import io
import os
import shutil

from PIL import Image, ImageOps, UnidentifiedImageError
import pytesseract
from pytesseract import TesseractError


MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 30_000_000


class ImageOCRError(RuntimeError):
    pass


def configure_tesseract():
    executable = pytesseract.pytesseract.tesseract_cmd
    if shutil.which(executable) or os.path.isfile(executable):
        return

    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return

    raise ImageOCRError(
        "Tesseract OCR is not installed or not on PATH. Install Tesseract OCR, "
        "restart PowerShell, and run the app again."
    )


def extract_review_text(image_bytes):
    if not image_bytes:
        raise ImageOCRError("The uploaded image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ImageOCRError("Image exceeds the 10 MB size limit.")

    configure_tesseract()

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ImageOCRError("Image resolution exceeds the 30 megapixel limit.")
            image = ImageOps.exif_transpose(source).convert("RGB")
    except ImageOCRError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ImageOCRError("Could not read this image. Try a valid PNG, JPG, or WebP file.") from error

    try:
        text = pytesseract.image_to_string(image, lang="eng", config="--psm 6")
    except pytesseract.TesseractNotFoundError as error:
        raise ImageOCRError(
            "Tesseract OCR executable was not found. Install Tesseract OCR and restart the app."
        ) from error
    except TesseractError as error:
        raise ImageOCRError(f"Tesseract could not process this image: {error}") from error

    return text.strip()
