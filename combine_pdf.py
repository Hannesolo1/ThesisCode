from pdf2image import convert_from_path
from PIL import Image

def combine_pdfs_side_by_side(pdf_paths, output_pdf_path):
    # Convert each PDF to a list of images (pages)
    images = []
    for pdf_path in pdf_paths:
        images.append(convert_from_path(pdf_path))

    # Assume all PDFs have the same number of pages
    combined_images = []
    num_pages = len(images[0])  # number of pages in the first PDF

    # Loop over each page of the PDFs and combine them side by side
    for i in range(num_pages):
        # Combine images of the same page across all PDFs
        page_images = [img[i] for img in images]

        # Create a new blank image with enough width to hold all pages side by side
        total_width = sum([img.width for img in page_images])
        max_height = max([img.height for img in page_images])

        # Create the combined image
        new_image = Image.new("RGB", (total_width, max_height))

        # Paste each page's image into the new image side by side
        current_x = 0
        for img in page_images:
            new_image.paste(img, (current_x, 0))
            current_x += img.width

        combined_images.append(new_image)

    # Save the combined images as a PDF
    combined_images[0].save(output_pdf_path, save_all=True, append_images=combined_images[1:])

# Example usage
pdf_paths = [ 'subgraph_visualization/leiden_images/subgraph_4_3.pdf', 'subgraph_visualization/kmeans_images/subgraph_4_3.pdf', 'subgraph_visualization/SAF_images/subgraph_4_3.pdf', 'subgraph_visualization/random_labels_images/subgraph_4_3.pdf']
output_pdf_path = 'subgraph_visualization/algos_same_subgraph0_3.pdf'

combine_pdfs_side_by_side(pdf_paths, output_pdf_path)
