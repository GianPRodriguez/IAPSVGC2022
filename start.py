import tkinter as tk
from tkinter import filedialog
import run
import subprocess

def Start(event=None):
	subprocess.call(["python", "run.py"])


root = tk.Tk()
root.geometry("500x309")
button = tk.Button(root, text='Open', command=Start)
button.pack()

root.mainloop()