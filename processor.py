from extractors import (
    extract_text_pdf,
    extract_text_txt,
    extract_text_docx,
    extract_text_pptx,
    extract_text_excel,
    extract_text_image,
)
from utils import save_temp_file


def process_uploaded_file(uploaded_file):
    """
    Recibe un archivo de Streamlit y devuelve el texto extraído.
    """

    extension = uploaded_file.name.lower()

    # Guardar archivo temporalmente
    path = save_temp_file(uploaded_file)

    # PDF
    if extension.endswith(".pdf"):
        return extract_text_pdf(path)

    # TXT
    elif extension.endswith(".txt"):
        return extract_text_txt(uploaded_file)

    # DOCX
    elif extension.endswith(".docx"):
        return extract_text_docx(path)

    # PPTX
    elif extension.endswith(".pptx"):
        return extract_text_pptx(path)

    # EXCEL (xls o xlsx)
    elif extension.endswith(".xls") or extension.endswith(".xlsx"):
        return extract_text_excel(path)

    # Imágenes
    elif extension.endswith((".png", ".jpg", ".jpeg", ".bmp")):
        return extract_text_image(path)

    else:
        return "❌ Tipo de archivo no soportado."
