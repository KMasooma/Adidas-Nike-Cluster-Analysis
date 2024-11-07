# Market Segmentation with KMeans Clustering: Adidas vs Nike

## Project Overview

This project explores market trends in Adidas and Nike products by using **Cluster Analysis** and **KMeans Clustering** to segment product offerings and uncover insights about pricing strategies, customer preferences, and product availability. This analysis was conducted using both raw data, scaled data, and data transformed using **Principal Component Analysis (PCA)**.

## Key Insights

- **Product Pricing Trends**: Identified differences in **sale** and **listing prices** for both brands, highlighting the market positioning of Adidas and Nike.
- **Brand Groupings**: Segmented both Adidas and Nike products into meaningful clusters (e.g., Adidas Originals and Nike Sportswear).
- **Discounting Strategies**: Found that Adidas typically has larger margins between sale and listing prices, suggesting different promotional strategies compared to Nike.

## Technologies Used

- **Python**
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `seaborn`
- **Principal Component Analysis (PCA)**
- **KMeans Clustering**

## Approach

### Step 1: Data Preparation
The dataset used for analysis contains product information from both **Adidas** and **Nike**. We performed preprocessing steps such as scaling the data and transforming it with PCA to reduce dimensionality.

### Step 2: KMeans Clustering
KMeans clustering was performed to segment the data into clusters based on key features like **listing price**, **sale price**, etc. We experimented with different numbers of clusters (n=2, 3) and used the **silhouette score** to evaluate the clustering quality.

### Step 3: Visualizations
We visualized the clusters and cluster centroids to provide a clear understanding of how products were grouped. The cluster analysis was done for:

- Raw data
- Scaled data
- PCA-transformed data

### Step 4: Silhouette Analysis
Silhouette scores were computed to measure the quality of clustering for each dataset, allowing us to assess which configuration resulted in the most cohesive and well-separated clusters.

## Project File Structure

. ├── data/ │ └── dataset.csv # Raw dataset of Adidas vs Nike products ├── notebooks/ │ └── cluster_analysis.ipynb # Jupyter notebook with KMeans clustering and analysis ├── visuals/ │ └── cluster_plot.png # Cluster visualizations for the analysis ├── requirements.txt # Python package dependencies ├── README.md # Project documentation

bash
Copy code

## How to Run

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/adidas-nike-cluster-analysis.git
   cd adidas-nike-cluster-analysis
Install dependencies:

Install the required Python libraries by running:

bash
Copy code
pip install -r requirements.txt
Run the Jupyter notebook:

Launch the Jupyter notebook to view the analysis:

bash
Copy code
jupyter notebook notebooks/cluster_analysis.ipynb
Results
The clustering process revealed distinct market trends for Adidas and Nike, with notable differences in pricing strategies and product availability. These insights can help brands tailor their marketing strategies and optimize pricing based on customer preferences.

Silhouette Scores:
Raw Data: [score value]
Scaled Data (n=3 clusters): [score value]
Scaled Data (n=2 clusters): [score value]
PCA Data: [score value]
Conclusion
This project highlights how cluster analysis and dimensionality reduction (using PCA) can help brands identify key trends and refine their strategies. The use of silhouette scores ensures that the resulting clusters are meaningful and well-separated.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Thanks to Subhasis Das for collaboration on the project.
The dataset used in this analysis was sourced from publicly available data (https://www.linkedin.com/safety/go?url=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fkaushiksuresh147%2Fadidas-vs-nike&trk=flagship-messaging-web&messageThreadUrn=urn%3Ali%3AmessagingThread%3A2-Mzg5NGFjOTUtNTc4YS00MDNhLThhYmQtYmRmMGQ3MjU2NzNkXzAxMA%3D%3D&lipi=urn%3Ali%3Apage%3Ad_flagship3_messaging_conversation_detail%3BrcMj3ek4SQy95Vm%2FY5NtFw%3D%3D).
