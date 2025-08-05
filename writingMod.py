import cv2
import numpy as np
import os
import glob
# For image drawing
from PIL import Image, ImageDraw, ImageFont, ImageEnhance


def change_contrast(img, level):
    # Input is a Image type
    # use Image.fromarray() on numpy
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)


def change_sharpness(img, level):
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(level)
    return img


def adjust(img, blevel, slevel, clevel, colevel):
    # brightness
    benhance = ImageEnhance.Brightness(img)
    img = benhance.enhance(blevel)
    # sharpness
    img = change_sharpness(img, slevel)
    # contrast
    img = change_contrast(img, clevel)
    # color
    cenhance = ImageEnhance.Color(img)
    img = cenhance.enhance(colevel)
    return img


def improveImage(img_dir):
    img = cv2.imread(img_dir, -1)
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, 
                                 beta=255, norm_type=cv2.NORM_MINMAX, 
                                 dtype=cv2.CV_8UC1)

        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return result_norm


def convert_image(image_dir):
    img = improveImage(image_dir)
    img = adjust(Image.fromarray(img), 
                  blevel=0.7, slevel=1, clevel=255, 
                  colevel=1)
    return img


def process_directory(input_dir, output_dir):
    """
    Process all images from input directory and save converted images to output directory
    
    Args:
        input_dir: Path to directory containing raw images to process
        output_dir: Path to directory where converted images will be saved
    
    Returns:
        Dictionary with processing results and statistics
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Created output directory: {output_dir}")
    
    # Supported image extensions
    supported_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    # Find all image files in input directory
    image_files = []
    for extension in supported_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, extension)))
        image_files.extend(glob.glob(os.path.join(input_dir, extension.upper())))
    
    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        return {"success": False, "message": "No image files found", "processed": 0, "failed": 0}
    
    print(f"🔍 Found {len(image_files)} images to process...")
    
    processed_count = 0
    failed_count = 0
    failed_files = []
    
    for image_path in image_files:
        try:
            # Get filename without extension
            filename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            # Convert the image
            print(f"🔄 Processing: {filename}")
            converted_img = convert_image(image_path)
            
            # Save converted image as JPG
            output_path = os.path.join(output_dir, f"{name_without_ext}.jpg")
            converted_img.save(output_path, "JPEG", quality=95)
            
            processed_count += 1
            print(f"✅ Saved: {output_path}")
            
        except Exception as e:
            failed_count += 1
            failed_files.append(filename)
            print(f"❌ Failed to process {filename}: {str(e)}")
    
    # Print summary
    print(f"\n📊 PROCESSING SUMMARY:")
    print(f"✅ Successfully processed: {processed_count} images")
    print(f"❌ Failed to process: {failed_count} images")
    
    if failed_files:
        print(f"📝 Failed files: {', '.join(failed_files)}")
    
    return {
        "success": True,
        "processed": processed_count,
        "failed": failed_count,
        "failed_files": failed_files,
        "output_directory": output_dir
    }


def convert_single_image_for_testing(image_path, output_dir=None):
    """
    Convert a single image for model testing
    
    Args:
        image_path: Path to the image to convert
        output_dir: Directory to save converted image (optional)
    
    Returns:
        Path to converted image
    """
    try:
        # Convert the image
        converted_img = convert_image(image_path)
        
        # Determine output path
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{name_without_ext}.jpg")
        else:
            # Save in same directory as input
            directory = os.path.dirname(image_path)
            filename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(filename)[0]
            output_path = os.path.join(directory, f"{name_without_ext}.jpg")
        
        # Save converted image
        converted_img.save(output_path, "JPEG", quality=95)
        print(f"✅ Converted image saved: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Failed to convert {image_path}: {str(e)}")
        return None