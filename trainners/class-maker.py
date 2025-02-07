import os
import tkinter as tk
from tkinter import filedialog

def select_folder():
    # Hide the root window
    root = tk.Tk()
    root.withdraw()

    # Open the folder selection dialog
    folder_path = filedialog.askdirectory(title="amharic_dataset")

    return folder_path

def get_subfolder_names(folder_path):
    # Get a list of all entries in the folder
    entries = os.listdir(folder_path)

    # Filter out only the subfolders
    subfolders = [entry for entry in entries if os.path.isdir(os.path.join(folder_path, entry))]

    return subfolders

if __name__ == "__main__":
    # Select a folder
    selected_folder = select_folder()

    if selected_folder:
        # Get the names of subfolders
        subfolder_names = get_subfolder_names(selected_folder)

        # Print the array of subfolder names
        print(subfolder_names)
    else:
        print("No folder selected.")