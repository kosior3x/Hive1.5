
import os
from pypdf import PdfReader

def extract_single_pdf(filename):
    print(f"Extracting {filename}...")
    try:
        reader = PdfReader(filename)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        output_filename = "protocol_analysis_content.txt"
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Extraction complete. Saved to {output_filename}")

    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    extract_single_pdf("Praca nad agentem rojem świadomości - tenatyka i rowazania nad konceptem protokołu kom caotycznych neuronów.pdf")
