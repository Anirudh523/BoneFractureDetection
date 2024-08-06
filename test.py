file_path = r'C:\Users\Anirudh\Documents\GitHub\BoneFractureDetection\BoneFractureData2\BoneFractureYolo8\train\images'  # Replace this with the path to your file
labels_path = r'C:\Users\Anirudh\Documents\GitHub\BoneFractureDetection\BoneFractureData2\BoneFractureYolo8\train\labels'
try:
    print(len(file_path))
    print(len(labels_path))
except FileNotFoundError:
    print(f"FileNotFoundError: The file at '{file_path}' does not exist.")
except PermissionError:
    print(f"PermissionError: You do not have permission to access the file at '{file_path}'.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

