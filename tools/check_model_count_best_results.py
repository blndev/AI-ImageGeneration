import os
import re
from dotenv import load_dotenv

# after review and moving accepted images into a eval folder, the amount of images per model will be counted and the top 20 listed

def count_file_parts(directory):
    model_count = {}
    model_pattern = re.compile(r'_(?!prompt)([a-z][\w-]*\.safetensors)\b')

    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            # Split the filename into parts
            models = model_pattern.findall(file)  # file.split('_')
            #if len(models) > 0: print(models)
            for model in models:
                # Increment the count for the part
                if model in model_count:
                    model_count[model] += 1
                else:
                    model_count[model] = 1

    # Convert the dictionary to a sorted list of tuples
    sorted_model_by_count = dict(sorted(model_count.items(), key=lambda item: item[1], reverse=True))
    l = list(sorted_model_by_count.items())
    top20 = l[:20]
    return top20


if __name__ == "__main__":
    try:
        load_dotenv(override=True)
        # Specify the directory to scan
        directory_to_scan = os.getenv('RESULT_DIRECTORY', './output')
        print(f"Start scanning {directory_to_scan}")
        sorted_parts_count = count_file_parts(directory_to_scan)

        for model, value in sorted_parts_count:
            print(f"{model}: {value}")
    except KeyboardInterrupt:
        print("Shutdown")
        exit()
