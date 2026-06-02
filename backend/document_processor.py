import fitz
from docx import Document
import os


def extract_pdf_text(filepath):
    text = ""

    pdf = fitz.open(filepath)

    for page in pdf:
        text += page.get_text()

    return text


def extract_docx_text(filepath):
    doc = Document(filepath)

    text = "\n".join(
        paragraph.text
        for paragraph in doc.paragraphs
    )

    return text


def extract_text(filepath):

    extension = os.path.splitext(filepath)[1].lower()

    if extension == ".pdf":
        return extract_pdf_text(filepath)

    elif extension == ".docx":
        return extract_docx_text(filepath)

    elif extension == ".txt":
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )