"""Generate sample_policy.pdf from sample_policy.md for testing the RAG pipeline."""

import os

try:
    from fpdf import FPDF
except ImportError:
    print("Install fpdf2: pip install fpdf2")
    raise SystemExit(1)


def md_to_pdf(md_path: str, pdf_path: str):
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    for line in lines:
        stripped = line.rstrip("\n")

        if stripped.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, stripped[2:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)
        elif stripped.startswith("## "):
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 9, stripped[3:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
        elif stripped.startswith("### "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, stripped[4:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif stripped.startswith("---"):
            pdf.ln(4)
        elif stripped.startswith("|"):
            pdf.set_font("Courier", "", 9)
            pdf.cell(0, 5, stripped, new_x="LMARGIN", new_y="NEXT")
        elif stripped.startswith("- "):
            pdf.set_font("Helvetica", "", 11)
            pdf.set_x(pdf.l_margin + 8)
            pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 8, 6, f"- {stripped[2:]}")
        elif stripped.strip() and stripped.strip()[0].isdigit() and ". " in stripped[:4]:
            pdf.set_font("Helvetica", "", 11)
            pdf.set_x(pdf.l_margin + 8)
            pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 8, 6, stripped.strip())
        elif stripped.strip() == "":
            pdf.ln(3)
        else:
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, stripped)

    pdf.output(pdf_path)
    print(f"Created: {pdf_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(base_dir, "docs", "sample_policy.md")
    pdf_path = os.path.join(base_dir, "docs", "sample_policy.pdf")
    md_to_pdf(md_path, pdf_path)
