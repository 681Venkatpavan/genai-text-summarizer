from flask import Flask, render_template, request
from transformers import pipeline
import fitz  # PyMuPDF
import os

app = Flask(__name__)

# Folder to store uploaded PDFs
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load summarization model (T5)
summarizer = pipeline("summarization", model="t5-small")


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text


@app.route("/", methods=["GET", "POST"])
def home():
    expert_summary = ""
    simplified_summary = ""
    key_insights = ""
    original_text = ""

    if request.method == "POST":

        # Get text from textarea
        text_input = request.form.get("text")

        # Get uploaded PDF
        pdf_file = request.files.get("pdf")

        if pdf_file and pdf_file.filename != "":
            pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_file.filename)
            pdf_file.save(pdf_path)
            original_text = extract_text_from_pdf(pdf_path)

        elif text_input.strip() != "":
            original_text = text_input

        if original_text.strip() != "":
            # Limit long input (T5-small limit)
            original_text = original_text[:2000]

            # Expert Summary
            expert = summarizer(
                original_text,
                max_length=120,
                min_length=40,
                do_sample=False
            )
            expert_summary = expert[0]["summary_text"]

            # Simplified Summary
            simple = summarizer(
                original_text,
                max_length=60,
                min_length=20,
                do_sample=False
            )
            simplified_summary = simple[0]["summary_text"]

            # Key Insights
            insights = summarizer(
                original_text,
                max_length=40,
                min_length=15,
                do_sample=False
            )
            key_insights = insights[0]["summary_text"]

    return render_template(
        "index.html",
        expert_summary=expert_summary,
        simplified_summary=simplified_summary,
        key_insights=key_insights,
    )


if __name__ == "__main__":
    app.run(debug=True)