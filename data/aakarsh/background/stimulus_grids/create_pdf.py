# create_pdf.py
# This script finds all 'stimulus_grid_page_*.png' files, sorts them numerically,
# and combines them into a single PDF document.

import os
import re
import glob
from PIL import Image

def create_pdf_from_images(image_files, output_pdf_path):
    """
    Creates a PDF from a list of image files.

    Args:
        image_files (list): A list of paths to the image files.
        output_pdf_path (str): The path for the output PDF file.
    """
    if not image_files:
        print("No image files found to process.")
        return

    # --- Step 1: Sort the files numerically based on the page number ---
    # We use a regular expression to find the number in each filename.
    def get_page_number(filename):
        match = re.search(r'_(\d+)\.png$', filename)
        if match:
            return int(match.group(1))
        return -1 # Return -1 if no number is found, for error handling

    # Sort the list using the extracted page number
    sorted_files = sorted(image_files, key=get_page_number)
    
    print("Found and sorted the following files:")
    for f in sorted_files:
        print(f" - {os.path.basename(f)}")

    # --- Step 2: Open images and prepare for PDF conversion ---
    image_list = []
    for filename in sorted_files:
        try:
            img = Image.open(filename)
            # Convert to RGB to ensure compatibility with PDF saving
            # This prevents potential errors with different image palettes (e.g., RGBA or P)
            img = img.convert('RGB')
            image_list.append(img)
        except Exception as e:
            print(f"Could not open or convert {filename}. Error: {e}")
            continue

    if not image_list:
        print("No valid images could be processed.")
        return

    # --- Step 3: Save the images into a single PDF ---
    # The first image is used to initialize the save, and the rest are appended.
    first_image = image_list[0]
    remaining_images = image_list[1:]

    try:
        first_image.save(
            output_pdf_path,
            "PDF",
            resolution=100.0,
            save_all=True,
            append_images=remaining_images
        )
        print(f"\nSuccessfully created PDF: {output_pdf_path}")
    except Exception as e:
        print(f"An error occurred while saving the PDF: {e}")


# --- Main execution block ---
if __name__ == "__main__":
    # --- Use glob to find all matching PNG files automatically ---
    # This makes the script more robust and easier to use.
    file_pattern = "stimulus_grid_page_*.png"
    png_files = glob.glob(file_pattern)

    if not png_files:
        print(f"No files found in the current directory matching the pattern: {file_pattern}")
    else:
        # Define the name of the output file
        output_pdf = "stimulus_grid_report.pdf"
        
        # Run the function to create the PDF
        create_pdf_from_images(png_files, output_pdf)

