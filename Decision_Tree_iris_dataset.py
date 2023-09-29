import tkinter as tk
from tkinter import ttk
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the Iris dataset as an example
iris = load_iris()
X, y = iris.data, iris.target

# Create a Decision Tree classifier
clf = DecisionTreeClassifier()

# Fit the classifier on the data
clf.fit(X, y)

# Create a Tkinter GUI
root = tk.Tk()
root.title("Decision Tree Classifier")

# Label
label = ttk.Label(root, text="Enter Sepal Length, Sepal Width, Petal Length, Petal Width:")
label.pack()

# Entry fields
entry_values = []
for i in range(4):
    entry = ttk.Entry(root)
    entry_values.append(entry)
    entry.pack()

# Predict button
def predict():
    input_data = [float(entry.get()) for entry in entry_values]
    prediction = clf.predict([input_data])
    result_label.config(text=f"Predicted Class: {iris.target_names[prediction[0]]}")

predict_button = ttk.Button(root, text="Predict", command=predict)
predict_button.pack()

# Result label
result_label = ttk.Label(root, text="")
result_label.pack()

# Create a figure for the decision tree visualization
fig, ax = plt.subplots(figsize=(5, 5))
plt.axis('off')

# Visualize the decision tree
from sklearn.tree import plot_tree
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
figure_canvas = FigureCanvasTkAgg(fig, master=root)
figure_canvas_widget = figure_canvas.get_tk_widget()
figure_canvas_widget.pack()

root.mainloop()