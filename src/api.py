from fastapi import FastAPI, File, UploadFile
import fitz
from inference import TableExtractionPipeline
from pdf2image import convert_from_bytes
from PIL import Image


def is_pdf(file_name):
    return file_name.lower().endswith(".pdf")


def get_tokens_from_pdf_page(page, dpi=100):
    flags = fitz.TEXT_INHIBIT_SPACES & ~fitz.TEXT_PRESERVE_IMAGES

    words = page.get_text(option="words", flags=flags)

    # converting 'words' to a list of dicts instead of a list of tuples
    tokens = []
    for word_meta in words:
        tokens.append(
            {
                # times (dpi / 72) is to make sure the bounding boxes are in the same scale as the generated image
                "bbox": list(
                    map(
                        lambda x: int(x * (dpi / 72)),
                        [word_meta[0], word_meta[1], word_meta[2], word_meta[3]],
                    )
                ),
                "text": word_meta[4],
                "flags": 0,
                "block_num": word_meta[5],
                "line_num": word_meta[6],
                "span_num": word_meta[7],
            }
        )

    return tokens


pipe = TableExtractionPipeline(
    det_config_path="detection_config.json",
    det_model_path="../pubtables1m_detection_detr_r18.pth",
    det_device="cuda",
    str_config_path="structure_config.json",
    str_model_path="../pubtables1m_structure_detr_r18.pth",
    str_device="cuda",
)

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    # print(f"Received file: {file.filename}")
    try:
        if not is_pdf(file.filename):
            return {"message": "Only PDF files are allowed for upload"}

        html = ""
        pdf_bytes = file.file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = pdf_document.page_count

        for i, page in enumerate(pdf_document.pages()):
            tokens = get_tokens_from_pdf_page(page)

            images = convert_from_bytes(
                pdf_bytes, first_page=i, last_page=i + 1, dpi=100
            )
            img = images[0]

            extracted_tables = pipe.recognize(
                img,
                tokens,
                out_objects=True,
                out_cells=True,
                out_html=True,
                out_csv=True,
            )

            for table in extracted_tables["html"]:
                html += table

    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}
    finally:
        file.file.close()

    return {"html": html}
