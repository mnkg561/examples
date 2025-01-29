
Create a Python program for Kubernetes HPA optimization that analyzes historical metrics and recommends optimal settings. The program should run in a Jupyter notebook environment.

Input Parameters:
- Namespace
- Deployment name
- Target CPU utilization threshold (e.g., 45%)

Data Collection (14 days historical):
1. Time series data:
   - CPU utilization per timestamp
   - Replica count at each timestamp
   - Pod CPU requests (cores)

Analysis Requirements:

Basic Analysis:
For each timestamp in the 14-day period:
Calculate actual CPU cores used = replicas * CPU request * utilization%
Store the relationship between load and replica count

Statistical Analysis:
Calculate:
P95 CPU utilization across all timestamps
Daily/weekly patterns using time series decomposition
Correlation between load and replica count

Optimization Logic:
A. If P95 > input target utilization:
   
Use P95 as new target (capped at standard threshold)
Recalculate minimum replicas based on P95

B. For minimum replica calculation:
   
Find minimum CPU cores needed across all timestamps
Calculate: min_replicas = ceil(min_cores / (CPU_request * new_target))

Optional ML Components:
Use Prophet or SARIMA for load pattern analysis
Cluster analysis to identify usage patterns
Anomaly detection to exclude outlier periods
Regression analysis for replica vs load relationship

Output:
1. Recommendations:
   - New target CPU utilization (original or P95-based)
   - Recommended minimum replicas
   - Confidence score based on data consistency

Supporting Analysis:
Usage pattern graphs
Statistical validation of recommendations
Resource savings projections
Pattern analysis results
Example Calculation:
Given a 14-day period where:
- Lowest period shows: 70 cores total usage (100 replicas * 10 cores * 7% util)
- P95 utilization is 38% (higher than input target of 25%)
- Pod CPU request is 10 cores

Then:
1. New target = 38% (P95 instead of input 25%)
2. Min replicas = ceil(70 / (10 * 0.38)) = 2

Required Python Libraries:
- prometheus_api_client
- pandas
- numpy
- scipy
- prophet (optional for ML)
- statsmodels
- scikit-learn
- matplotlib/plotly for visualizations

The program should be modular to allow for:
1. Easy switching between different analysis methods
2. Addition of new optimization algorithms
3. Integration of different data sources
4. Custom metric combinations

Notes:
- All calculations should consider the full 14-day history
- Recommendations should be validated against multiple time windows
- Include confidence scores based on data consistency
- Generate visualizations to support
