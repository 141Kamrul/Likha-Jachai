#!/usr/bin/env python3
"""
Test script to run writingMod.py functions
"""

from writingMod import process_directory, convert_single_image_for_testing, convert_image
import os

def main():
    print("🚀 Testing writingMod.py functions")
    print("=" * 50)
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Example 1: Test single image conversion
    print("\n📋 Example 1: Convert single image from converted folder")
    converted_dir = os.path.join(current_dir, "raw_test_handwriting")
    
    if os.path.exists(converted_dir):
        # Find first image in converted folder
        image_files = [f for f in os.listdir(converted_dir) if f.endswith('.jpg')]
        if image_files:
            test_image = os.path.join(converted_dir, image_files[0])
            print(f"Testing with: {test_image}")
            
            # Convert single image
            output_path = convert_single_image_for_testing(
                test_image, 
                output_dir="converted_test_output"
            )
            if output_path:
                print(f"✅ Successfully converted: {output_path}")
            else:
                print("❌ Conversion failed")
        else:
            print("❌ No images found in converted folder")
    else:
        print("❌ Converted folder not found")
    
    # Example 2: Test batch processing
    print("\n📋 Example 2: Batch process directory")
    
    # You can replace these paths with your actual input/output directories
    input_dir = "your_input_images"  # Replace with your folder path
    output_dir = "batch_converted_output"
    
    print(f"To batch process images:")
    print(f"1. Create a folder: {input_dir}")
    print(f"2. Put your raw images in that folder")
    print(f"3. Run: process_directory('{input_dir}', '{output_dir}')")
    
    # Uncomment these lines when you have input images:
    # if os.path.exists(input_dir):
    #     result = process_directory(input_dir, output_dir)
    #     print(f"Batch processing result: {result}")
    # else:
    #     print(f"Create folder '{input_dir}' and add images to test batch processing")

if __name__ == "__main__":
    main()
