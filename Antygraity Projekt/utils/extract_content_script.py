
import os
from pypdf import PdfReader


def extract_text_from_files(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    output_path = os.path.join(directory, "extracted_content_full.txt")

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for filename in files:
            if filename == "extracted_content_full.txt" or filename == "extract_content_script.py":
                continue

            filepath = os.path.join(directory, filename)
            outfile.write(f"\n{'='*50}\nFile: {filename}\n{'='*50}\n")

            try:
                if filename.lower().endswith('.pdf'):
                    try:
                        reader = PdfReader(filepath)
                        text = ""
                        for page in reader.pages:
                            extracted = page.extract_text()
                            if extracted:
                                text += extracted + "\n"
                        outfile.write(text)
                    except Exception as pdf_err:
                        outfile.write(f"Error reading PDF: {pdf_err}\n")

                elif filename.lower().endswith('.txt'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        outfile.write(f.read())
                else:
                    outfile.write("[Skipping non-text/pdf file]\n")

            except Exception as e:
                outfile.write(f"Error reading file {filename}: {e}\n")
    print(f"Extraction complete. Saved to {output_path}")

if __name__ == "__main__":
    extract_text_from_files(".")
