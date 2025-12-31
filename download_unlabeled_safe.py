from roboflow import Roboflow
import os
import requests

# --- CONFIGURATION ---
API_KEY = "M39u3k4RslacoYdEbhQB"  # PASTE YOUR KEY HERE
PROJECT_ID = "robotic-arm-0r333"
WORKSPACE_ID = "label-nhdaa"
DOWNLOAD_FOLDER = "unlabeled_images"

# Initialize
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)

if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

print("Scanning project for unlabeled images... (Safe Mode)")

# 1. Fetch all images using search_all (Handle Pagination)
#    We use a big limit to ensure we get pages
page_generator = project.search_all(limit=1000)

count = 0
downloaded = 0

try:
    for page in page_generator:
        # SAFETY CHECK: Is 'page' a list of images or a single image dict?
        # The SDK sometimes yields lists (pages) and sometimes dicts (items).
        if isinstance(page, list):
            iterable = page
        else:
            iterable = [page]

        for image in iterable:
            count += 1
            
            # 2. SAFETY CHECK: Find the Name
            #    Try every possible key name Roboflow might use
            image_name = image.get('name') or image.get('filename') or image.get('original_filename')
            image_id = image.get('id')

            # Fallback: If no name found, make one up using ID
            if not image_name:
                if image_id:
                    image_name = f"{image_id}.jpg"
                else:
                    print(f"Skipping corrupt record (No ID/Name)")
                    continue

            # 3. SAFETY CHECK: Check Annotation Status
            #    We only want images with NO annotations
            is_unlabeled = False
            
            # Check 'annotation' key (Object or None)
            if image.get('annotation') is None:
                is_unlabeled = True
            
            # Check 'annotations' key (List or Empty)
            elif 'annotations' in image:
                if len(image['annotations']) == 0:
                    is_unlabeled = True
            
            # If valid unlabeled image, download it
            if is_unlabeled:
                url = f"https://source.roboflow.com/{WORKSPACE_ID}/{PROJECT_ID}/{image_id}/original.jpg?api_key={API_KEY}"
                
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        file_path = os.path.join(DOWNLOAD_FOLDER, image_name)
                        with open(file_path, "wb") as f:
                            f.write(response.content)
                        downloaded += 1
                        if downloaded % 50 == 0:
                            print(f"Downloaded {downloaded} images...")
                    else:
                        # 404 means the image might be processing still
                        pass
                except Exception as e:
                    print(f"Warning: Could not download {image_name}")

except Exception as e:
    print(f"\nCRITICAL ERROR: {e}")
    print("However, check your folder - we may have downloaded some images already.")

print(f"✅ DONE. Scanned {count} records. Downloaded {downloaded} unlabeled images.")