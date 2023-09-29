import tkinter as tk
from tkinter import ttk
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the Diabetes dataset as an example
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Create a Decision Tree Regressor
clf = DecisionTreeRegressor()

# Fit the regressor on the data
clf.fit(X, y)

# Create a Tkinter GUI
root = tk.Tk()
root.title("Decision Tree Regressor for Diabetes Progression")

# Label
label = ttk.Label(root, text="Enter feature values:")
label.pack()

# Entry fields
entry_values = []
for i, feature_name in enumerate(diabetes.feature_names):
    entry_label = ttk.Label(root, text=feature_name)
    entry_label.pack()
    entry = ttk.Entry(root)
    entry_values.append(entry)
    entry.pack()

# Predict button
def predict():
    input_data = [float(entry.get()) for entry in entry_values]
    prediction = clf.predict([input_data])
    result_label.config(text=f"Predicted Diabetes Progression: {prediction[0]:.2f}")

predict_button = ttk.Button(root, text="Predict", command=predict)
predict_button.pack()

# Result label
result_label = ttk.Label(root, text="")
result_label.pack()

# Create a figure for the decision tree visualization (not applicable to Diabetes dataset)
# You can remove this part as the Diabetes dataset doesn't require a decision tree visualization.

root.mainloop()