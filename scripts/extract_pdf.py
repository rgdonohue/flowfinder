#!/usr/bin/env python3
"""
Extract content from PDF and convert to markdown format.
"""

import PyPDF2
import re
import sys
from pathlib import Path

def extract_pdf_to_markdown(pdf_path, output_path):
    """Extract text from PDF and convert to markdown format."""
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            markdown_content = []
            markdown_content.append(f"# Watershed Delineation Tools Competitive Analysis\n\n")
            markdown_content.append(f"*Extracted from: {pdf_path.name}*\n\n")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                
                if text.strip():
                    # Add page separator
                    markdown_content.append(f"## Page {page_num}\n\n")
                    
                    # Clean up the text
                    cleaned_text = clean_text(text)
                    
                    # Convert to markdown format
                    markdown_text = convert_to_markdown(cleaned_text)
                    markdown_content.append(markdown_text)
                    markdown_content.append("\n\n")
            
            # Write to output file
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(''.join(markdown_content))
                
            print(f"Successfully extracted PDF content to: {output_path}")
            return True
            
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return False

def clean_text(text):
    """Clean up extracted text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers and headers/footers
    text = re.sub(r'\b\d+\s*of\s*\d+\b', '', text)
    
    # Clean up line breaks
    text = text.replace('\n', ' ')
    
    return text.strip()

def convert_to_markdown(text):
    """Convert plain text to markdown format."""
    # Split into paragraphs
    paragraphs = text.split('  ')
    
    markdown_paragraphs = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph:
            # Try to detect headers (all caps or numbered)
            if re.match(r'^[A-Z\s]{5,}$', paragraph) or re.match(r'^\d+\.\s*[A-Z]', paragraph):
                markdown_paragraphs.append(f"### {paragraph}\n")
            else:
                markdown_paragraphs.append(f"{paragraph}\n\n")
    
    return ''.join(markdown_paragraphs)

if __name__ == "__main__":
    pdf_path = Path("docs/Watershed Delineation Tools Competitive Analysis.pdf")
    output_path = Path("docs/watershed_delineation_competitive_analysis.md")
    
    if not pdf_path.exists():
        print(f"PDF file not found: {pdf_path}")
        sys.exit(1)
    
    success = extract_pdf_to_markdown(pdf_path, output_path)
    if success:
        print("PDF extraction completed successfully!")
    else:
        print("PDF extraction failed!")
        sys.exit(1) 