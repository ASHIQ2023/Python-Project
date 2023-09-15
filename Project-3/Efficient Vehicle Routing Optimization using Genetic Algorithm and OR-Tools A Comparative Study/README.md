# Efficient Vehicle Routing Optimization using Genetic Algorithm and OR-Tools: A Comparative Study
This repository contains code for optimizing delivery routes using OR-Tools and visualizing the routes on a map using Folium.

# Table of Contents
Introduction<br>
Installation<br>
Usage<br>
Examples<br>


# Introduction
Delivery route optimization plays a important role in enhancing the efficiency of transportation activities. This repository provides a solution to the vehicle routing problem using OR-Tools, a powerful optimization library, and visualizes the optimized routes on an interactive map using Folium, a Python library for map visualization.

# Installation

To use the code in this repository, follow these steps:
1.	Clone the repository:
bashCopy code
git clone https://github.com/ASHIQ2023/Efficient-Vehicle-Routing-Optimization-using-Genetic-Algorithm-and-OR-Tools-A-Comparative-Study.git

3.	Download and prepare the input data:<br>
• Place the distance matrix file (distance.xlsx) in the project directory.<br>
•	Modify the file path in the code to match the location of the distance matrix file.<br>

# Usage
The main script optimize_routes.ipynb performs the optimization of delivery routes and generates an interactive map visualization.<br>
Make sure you have the necessary dependencies installed. You will need the following libraries: ortools, pandas, numpy, and folium. You can install them using pip:<br>
pip install ortools pandas numpy folium<br>

To run the script, execute the following command:<br>
Create a new Python file, e.g., delivery_route_optimization.py, and copy the code into it.<br>

Replace the file path in the line df_distance = pd.read_excel('C:\\Users\\ASHIQ\\Desktop\\distance.xlsx', index_col=0) with the correct path to your distance.xlsx file. Make sure the file exists and contains the required data.<br>

Save the file.<br>

Run the Python script using the command:<br>
python delivery_route_optimization.py<br>

The script uses the OR-Tools library to solve the vehicle routing problem, optimizing the routes based on the provided distance matrix and parcel demands. It then generates an HTML file (route_map.html) containing the interactive map visualization of the optimized routes.

# Examples

The repository includes an example distance matrix file (distance.xlsx) and parcel demand data. You can use this example data to test the code and observe the optimized routes.<br>

To run the example, follow the Installation and Usage instructions mentioned above.


