import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Sample RGB color dataset with more color labels
colors = {
    "Red": (255, 0, 0),
    "Green": (0, 255, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Purple": (128, 0, 128),
    "Orange": (255, 165, 0),
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Pink": (255, 182, 193),
    "Cyan": (0, 255, 255),
    "Brown": (165, 42, 42),
    "Gray": (128, 128, 128),
}

# Convert color dictionary to lists of RGB values and color labels
rgb_values = list(colors.values())
color_labels = list(colors.keys())

# Flatten the RGB values to a single list
X = [rgb for rgb in rgb_values]

# Create labels for the RGB values (for testing purposes)
X_labels = [f"RGB({rgb[0]}, {rgb[1]}, {rgb[2]})" for rgb in rgb_values]

# Create labels for the color classes
y = [color_labels.index(label) for label in color_labels]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the KNN classifier
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Create a function to predict and display the result
def predict_color():
    try:
        # Get the input values from the GUI
        r = int(r_entry.get())
        g = int(g_entry.get())
        b = int(b_entry.get())

        # Make a prediction
        prediction = knn.predict([[r, g, b]])

        # Update the result label
        result_label.config(text=f"Predicted Color: {color_labels[prediction[0]]}")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values for RGB.")

# Create the main application window
app = tk.Tk()
app.title("RGB Color Predictor")

# Create a notebook-style tabbed interface
notebook = ttk.Notebook(app)
notebook.pack(fill=tk.BOTH, expand=True)

# Create tabs
tab1 = ttk.Frame(notebook)
tab2 = ttk.Frame(notebook)
notebook.add(tab1, text="Predict")
notebook.add(tab2, text="About")

# Create labels and entry fields for input on Tab 1
r_label = ttk.Label(tab1, text="Red (0-255):")
r_entry = ttk.Entry(tab1)
g_label = ttk.Label(tab1, text="Green (0-255):")
g_entry = ttk.Entry(tab1)
b_label = ttk.Label(tab1, text="Blue (0-255):")
b_entry = ttk.Entry(tab1)

# Create a button to trigger prediction on Tab 1
predict_button = ttk.Button(tab1, text="Predict", command=predict_color)

# Create a label to display the result on Tab 1
result_label = ttk.Label(tab1, text="Predicted Color: ")

# Arrange widgets in Tab 1
r_label.grid(row=0, column=0)
r_entry.grid(row=0, column=1)
g_label.grid(row=1, column=0)
g_entry.grid(row=1, column=1)
b_label.grid(row=2, column=0)
b_entry.grid(row=2, column=1)
predict_button.grid(row=3, columnspan=2)
result_label.grid(row=4, columnspan=2)

# Add content to Tab 2 (About)
about_label = ttk.Label(tab2, text="This GUI demonstrates a KNN classifier for predicting colors based on RGB values.")
about_label.pack(padx=20, pady=20)

# Start the GUI application
app.mainloop()