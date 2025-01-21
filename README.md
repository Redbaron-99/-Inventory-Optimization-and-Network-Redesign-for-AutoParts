## Project: InventOptim - Inventory Optimization and Network Design System

### Project Overview

InventOptim is a Python-based system designed to optimize inventory and distribution networks for large-scale distributors. The project will use real-world-inspired data to simulate the operations of a national auto parts distributor, "AutoParts Express," with over 100 local Distribution Centers (DCs).

### Project Structure

```
InventOptim/
│
├── data/
│   ├── sales_data.csv
│   ├── product_data.csv
│   └── location_data.csv
│
├── src/
│   ├── __init__.py
│   ├── data_processor.py
│   ├── abc_analyzer.py
│   ├── network_optimizer.py
│   ├── inventory_optimizer.py
│   └── visualizer.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_processor.py
│   ├── test_abc_analyzer.py
│   ├── test_network_optimizer.py
│   └── test_inventory_optimizer.py
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── results_analysis.ipynb
│
├── requirements.txt
├── README.md
└── main.py
```

### Detailed Component Descriptions

1. **Data Processor (`data_processor.py`)**
   - Responsible for loading and cleaning the data from CSV files.
   - Implements data validation and error handling.

```python
import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, sales_file, product_file, location_file):
        self.sales_file = sales_file
        self.product_file = product_file
        self.location_file = location_file

    def load_data(self):
        self.sales_data = pd.read_csv(self.sales_file)
        self.product_data = pd.read_csv(self.product_file)
        self.location_data = pd.read_csv(self.location_file)

    def clean_data(self):
        # Remove duplicates
        self.sales_data.drop_duplicates(inplace=True)
        
        # Handle missing values
        self.sales_data.fillna(0, inplace=True)
        
        # Convert date columns to datetime
        self.sales_data['date'] = pd.to_datetime(self.sales_data['date'])

    def merge_data(self):
        self.merged_data = pd.merge(self.sales_data, self.product_data, on='product_id')
        self.merged_data = pd.merge(self.merged_data, self.location_data, on='location_id')

    def process_data(self):
        self.load_data()
        self.clean_data()
        self.merge_data()
        return self.merged_data
```

2. **ABC Analyzer (`abc_analyzer.py`)**
   - Performs ABC analysis on the product data.
   - Categorizes products based on their sales volume and revenue contribution.

```python
import pandas as pd
import numpy as np

class ABCAnalyzer:
    def __init__(self, data):
        self.data = data

    def calculate_revenue(self):
        self.data['total_revenue'] = self.data['quantity'] * self.data['price']

    def sort_by_revenue(self):
        return self.data.sort_values('total_revenue', ascending=False)

    def calculate_cumulative_percentage(self, sorted_data):
        sorted_data['cumulative_revenue'] = sorted_data['total_revenue'].cumsum()
        sorted_data['revenue_percentage'] = sorted_data['cumulative_revenue'] / sorted_data['total_revenue'].sum()

    def categorize(self, sorted_data):
        sorted_data['category'] = np.where(sorted_data['revenue_percentage'] <= 0.8, 'A',
                                           np.where(sorted_data['revenue_percentage'] <= 0.95, 'B', 'C'))

    def perform_abc_analysis(self):
        self.calculate_revenue()
        sorted_data = self.sort_by_revenue()
        self.calculate_cumulative_percentage(sorted_data)
        self.categorize(sorted_data)
        return sorted_data
```

3. **Network Optimizer (`network_optimizer.py`)**
   - Implements algorithms to determine optimal locations for Regional Distribution Centers (RDCs).
   - Uses linear programming to minimize total distribution costs.

```python
from pulp import *

class NetworkOptimizer:
    def __init__(self, locations, demands, costs):
        self.locations = locations
        self.demands = demands
        self.costs = costs

    def create_model(self):
        self.model = LpProblem("RDC_Location_Optimization", LpMinimize)

        # Decision variables
        self.x = LpVariable.dicts("open_rdc", self.locations, lowBound=0, upBound=1, cat='Binary')
        self.y = LpVariable.dicts("assign", [(i, j) for i in self.locations for j in self.demands.keys()], lowBound=0, upBound=1, cat='Binary')

        # Objective function
        self.model += lpSum([self.costs[i]['setup'] * self.x[i] for i in self.locations]) + \
                      lpSum([self.costs[i][j] * self.demands[j] * self.y[(i, j)] for i in self.locations for j in self.demands.keys()])

    def add_constraints(self):
        # Each DC must be assigned to exactly one RDC
        for j in self.demands.keys():
            self.model += lpSum([self.y[(i, j)] for i in self.locations]) == 1

        # If a DC is assigned to an RDC, that RDC must be open
        for i in self.locations:
            for j in self.demands.keys():
                self.model += self.y[(i, j)] <= self.x[i]

        # Maximum number of RDCs constraint
        self.model += lpSum([self.x[i] for i in self.locations]) <= 5  # Adjust as needed

    def solve_model(self):
        self.create_model()
        self.add_constraints()
        self.model.solve()

        results = {
            'status': LpStatus[self.model.status],
            'total_cost': value(self.model.objective),
            'open_rdcs': [loc for loc in self.locations if value(self.x[loc]) > 0.5],
            'assignments': {j: next(i for i in self.locations if value(self.y[(i, j)]) > 0.5) for j in self.demands.keys()}
        }

        return results
```

4. **Inventory Optimizer (`inventory_optimizer.py`)**
   - Implements multi-echelon inventory optimization algorithms.
   - Calculates optimal inventory levels and reorder points for each SKU at each location.

```python
import numpy as np
from scipy.stats import norm

class InventoryOptimizer:
    def __init__(self, demand_data, lead_times, service_level, holding_cost, order_cost):
        self.demand_data = demand_data
        self.lead_times = lead_times
        self.service_level = service_level
        self.holding_cost = holding_cost
        self.order_cost = order_cost

    def calculate_safety_stock(self, demand, lead_time):
        z = norm.ppf(self.service_level)
        return z * np.sqrt(lead_time) * demand.std()

    def calculate_reorder_point(self, demand, lead_time, safety_stock):
        return demand.mean() * lead_time + safety_stock

    def calculate_order_quantity(self, annual_demand):
        return np.sqrt((2 * annual_demand * self.order_cost) / self.holding_cost)

    def optimize_inventory(self):
        results = {}
        for sku, location in self.demand_data.keys():
            demand = self.demand_data[(sku, location)]
            lead_time = self.lead_times[(sku, location)]
            
            safety_stock = self.calculate_safety_stock(demand, lead_time)
            reorder_point = self.calculate_reorder_point(demand, lead_time, safety_stock)
            order_quantity = self.calculate_order_quantity(demand.sum())
            
            results[(sku, location)] = {
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'order_quantity': order_quantity
            }
        
        return results
```

5. **Visualizer (`visualizer.py`)**
   - Creates visualizations for data analysis and result presentation.
   - Generates charts and graphs to illustrate optimization outcomes.

```python
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self, data):
        self.data = data

    def plot_abc_distribution(self, abc_data):
        plt.figure(figsize=(10, 6))
        sns.barplot(x='category', y='total_revenue', data=abc_data)
        plt.title('ABC Analysis - Revenue Distribution')
        plt.xlabel('Category')
        plt.ylabel('Total Revenue')
        plt.show()

    def plot_inventory_levels(self, inventory_data):
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='location', y='inventory_level', data=inventory_data)
        plt.title('Inventory Levels by Location')
        plt.xlabel('Location')
        plt.ylabel('Inventory Level')
        plt.xticks(rotation=45)
        plt.show()

    def plot_network_map(self, locations, assignments):
        # This would typically use a mapping library like folium
        # For simplicity, we'll just print the assignments
        for dc, rdc in assignments.items():
            print(f"DC {dc} assigned to RDC {rdc}")

    def plot_cost_savings(self, before, after):
        labels = ['Before', 'After']
        costs = [before, after]
        plt.figure(figsize=(8, 6))
        plt.bar(labels, costs)
        plt.title('Cost Comparison Before and After Optimization')
        plt.ylabel('Total Cost')
        for i, v in enumerate(costs):
            plt.text(i, v, f'${v:,.0f}', ha='center', va='bottom')
        plt.show()
```

6. **Main Script (`main.py`)**
   - Orchestrates the entire optimization process.
   - Calls methods from other modules in the correct sequence.

```python
from src.data_processor import DataProcessor
from src.abc_analyzer import ABCAnalyzer
from src.network_optimizer import NetworkOptimizer
from src.inventory_optimizer import InventoryOptimizer
from src.visualizer import Visualizer

def main():
    # Initialize data processor and load data
    data_processor = DataProcessor('data/sales_data.csv', 'data/product_data.csv', 'data/location_data.csv')
    merged_data = data_processor.process_data()

    # Perform ABC analysis
    abc_analyzer = ABCAnalyzer(merged_data)
    abc_results = abc_analyzer.perform_abc_analysis()

    # Optimize network
    # (You would need to prepare the necessary inputs for the NetworkOptimizer here)
    network_optimizer = NetworkOptimizer(locations, demands, costs)
    network_results = network_optimizer.solve_model()

    # Optimize inventory
    # (You would need to prepare the necessary inputs for the InventoryOptimizer here)
    inventory_optimizer = InventoryOptimizer(demand_data, lead_times, service_level, holding_cost, order_cost)
    inventory_results = inventory_optimizer.optimize_inventory()

    # Visualize results
    visualizer = Visualizer(merged_data)
    visualizer.plot_abc_distribution(abc_results)
    visualizer.plot_inventory_levels(inventory_results)
    visualizer.plot_network_map(network_results['open_rdcs'], network_results['assignments'])
    visualizer.plot_cost_savings(before_cost, network_results['total_cost'])

if __name__ == "__main__":
    main()
```

### Additional Components

1. **Requirements File (`requirements.txt`)**
   List all necessary Python packages:
   ```
   pandas==1.3.3
   numpy==1.21.2
   scipy==1.7.1
   pulp==2.5.1
   matplotlib==3.4.3
   seaborn==0.11.2
   ```

2. **README File (`README.md`)**
   Provide a comprehensive description of the project, installation instructions, and usage guidelines.

3. **Test Files**
   Create unit tests for each module to ensure code reliability and facilitate maintenance.

### GitHub Upload Instructions

1. Initialize a new Git repository in your project folder:
   ```
   git init
   ```

2. Add all files to the repository:
   ```
   git add .
   ```

3. Commit the files:
   ```
   git commit -m "Initial commit of InventOptim project"
   ```

4. Create a new repository on GitHub.

5. Link your local repository to the GitHub repository:
   ```
   git remote add origin https://github.com/yourusername/InventOptim.git
   ```

6. Push your code to GitHub:
   ```
   git push -u origin master
   ```

This project structure and implementation will provide a solid foundation for an inventory optimization and network design system. It demonstrates your skills in data analysis, optimization algorithms, and software development practices. The modular design allows for easy expansion and modification, making it an excellent portfolio piece for your GitHub profile.
