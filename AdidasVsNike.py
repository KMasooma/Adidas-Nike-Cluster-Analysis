#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats


# In[2]:


df=pd.read_csv("C:\\Users\\Masooma\\Adidas Vs Nike.csv")
df


# ## Data Understanding

# In[3]:


df.info()


# In[4]:


df.isna().sum()


# In[5]:


# Description column has 3 missing values


# In[6]:


df[df['Description'].isna()]


# In[7]:


def rows_Wo_descrip(df,product_name):
    rows=df[(df['Product Name']=='Nike React Infinity Run Flyknit') | (df['Product Name']=='Nike Free X Metcon 2')]
    return rows
rows_Wo_descrip(df,'Product Name')


# In[8]:


# So, we dont have description available for 'Nike React Infinity Run Flyknit'


# In[9]:


df.columns


# ## EDA

# In[10]:


df.describe()


# In[11]:


df1=df.copy()


# In[12]:


df1 = df1.drop(['Product ID', 'Description'], axis=1) # Dropping unnecessary columns


# In[13]:


# Checking for Zero Values in Key Columns

# Zeroes in columns like Listing Price or Sale Price may be placeholders for missing or unavailable data.

print(df1[df1['Listing Price'] == 0])
df1[df1['Sale Price'] == 0]


# In[14]:


df1[(df1['Listing Price']==0) & (df1['Discount']!=0)]


# In[15]:


# '0' Listing price may indiacate that these products are not available for sale or they are same as sale price (if they correspond to discount=0) 
#sale price has NO '0' values.
#'0' Listing price will be imputed by corresponding Sale price


# In[16]:


df1['Listing Price']=df1.apply(lambda row:row['Sale Price'] if row['Listing Price']==0 else row['Listing Price'], axis=1)
df1


# In[17]:


df1[((df1['Listing Price']-df1['Sale Price'])!=0) & (df1['Discount']==0)]


# In[18]:


# it seems some of the entries in 'Discount Column ' are wrongly entered as '0'. 
# So, its better to delete this column and recalculate it.


# In[19]:


df1=df1.drop(['Discount'],axis=1)

# Calculating Discount Percentage, handling zero values
df1['Discount % age'] = np.where(
    df1['Listing Price'] > 0,  # Condition: Listing Price must be greater than 0
    ((df1['Listing Price'] - df1['Sale Price']) / df1['Listing Price']) * 100,  # Discount calculation
    np.nan  # Assigning NaN for products with 0 listing price
)
df1


# ## Date_Time conversion 

# In[20]:


df1['Last Visited']=pd.to_datetime(df1['Last Visited'])


# #### Feature Engineering

# In[21]:


#creating a new categorical column indicating whether a product is "Available" or "Not Available" based on the listing price. 

df1['Availability'] = np.where(df1['Listing Price'] > 0, 'Available', 'Not Available')
df1


# # Brand Analysis

# In[22]:


df1['Brand'].unique()


# In[23]:


df1['Brand']=df1['Brand'].replace('Adidas Adidas ORIGINALS','Adidas ORIGINALS')


# In[24]:


df1['Product Name'].nunique()


# In[25]:


Adidas_prod=df1[df1['Brand']!= 'Nike']
Top5_Adidas_prod_count=Adidas_prod['Product Name'].value_counts().head(10)
Top5_Adidas_prod_count


# In[ ]:





# In[26]:


Nike_Prod=df1[df1['Brand']=='Nike']
Top5_Nike_prod_count=Nike_Prod['Product Name'].value_counts().head(10)
Top5_Nike_prod_count


# In[27]:


df1['Merged_var_brand']=np.where(df1['Brand']!='Nike','Adidas','Nike')
df1


# In[28]:


df1.columns


# #### Common Adidas & Nike Products 

# In[29]:


Adidas_prod=df1[df1['Merged_var_brand']!= 'Nike']
Top5_Adidas_prod_count=Adidas_prod['Product Name'].value_counts().head(5)
Top5_Adidas_prod_count


# In[30]:


Nike_prod=df1[df1['Merged_var_brand']=='Nike']
Top5_Nike_prod=Nike_prod['Product Name'].value_counts().head()
Top5_Nike_prod


# In[31]:


# Set up the figure size for two plots side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

# Plot for Top 5 Adidas Products (displayed first)
sns.set_theme(style='whitegrid')
sns.barplot(ax=axes[0], x=Top5_Adidas_prod_count.index, y=Top5_Adidas_prod_count.values)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)
axes[0].set_title('Most Common Adidas Products')

# Plot for Top 5 Nike Products (displayed second)
sns.barplot(ax=axes[1], x=Top5_Nike_prod.index, y=Top5_Nike_prod.values)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)
axes[1].set_title('Most Common Nike Products')

# Show the combined plot
plt.tight_layout()
plt.show()


# #### Rating by Brand & corresponding products

# In[32]:


# Calculate the average rating for each product within each brand
avg_ratings = df1.groupby(["Merged_var_brand", "Product Name"])["Rating"].mean().reset_index()

# Get the top 5 products by rating for each brand
top5_adidas = avg_ratings[avg_ratings["Merged_var_brand"] == "Adidas"].nlargest(10, "Rating")
top5_nike = avg_ratings[avg_ratings["Merged_var_brand"] == "Nike"].nlargest(10, "Rating")

# Set up the plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Adidas top 5 product ratings
sns.barplot(data=top5_adidas, x="Product Name", y="Rating", ax=axes[0], palette="Blues_d")
axes[0].set_title("Top 5 Adidas Product Ratings")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")

# Nike top 5 product ratings
sns.barplot(data=top5_nike, x="Product Name", y="Rating", ax=axes[1], palette="Reds_d")
axes[1].set_title("Top 5 Nike Product Ratings")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")

# Set common title and adjust layout
fig.suptitle("Top 5 Product Ratings by Brand")
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
plt.show()



# In[ ]:





# #### Listing & Sale Price by Brand

# In[33]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot for Listing Price
sns.histplot(df1[df1["Merged_var_brand"] == "Adidas"]["Listing Price"], kde=False, label='Adidas', ax=axes[0])
sns.histplot(df1[df1["Merged_var_brand"] == "Nike"]["Listing Price"], kde=False, label='Nike', ax=axes[0])
axes[0].legend()
axes[0].set_title('Listing Price Distribution')

# Plot for Sale Price
sns.histplot(df1[df1["Merged_var_brand"] == "Adidas"]["Sale Price"], kde=False, label='Adidas', ax=axes[1])
sns.histplot(df1[df1["Merged_var_brand"] == "Nike"]["Sale Price"], kde=False, label='Nike', ax=axes[1])
axes[1].legend()
axes[1].set_title('Sale Price Distribution')

plt.tight_layout()
plt.show()



# In[34]:


# It appears that Nike has low listing prices than those of Adidas.Same is true for Sale Prices as well.


# In[35]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Adidas plot: Sale and Listing Price
sns.histplot(df1[df1["Merged_var_brand"] == "Adidas"]["Listing Price"], kde=False, label='Listing Price', ax=axes[0])
sns.histplot(df1[df1["Merged_var_brand"] == "Adidas"]["Sale Price"], kde=False, label='Sale Price', ax=axes[0])
axes[0].legend()
axes[0].set_title('Adidas: Listing and Sale Price Distribution')

# Nike plot: Sale and Listing Price
sns.histplot(df1[df1["Merged_var_brand"] == "Nike"]["Listing Price"], kde=False, label='Listing Price', ax=axes[1])
sns.histplot(df1[df1["Merged_var_brand"] == "Nike"]["Sale Price"], kde=False, label='Sale Price', ax=axes[1])
axes[1].legend()
axes[1].set_title('Nike: Listing and Sale Price Distribution')

plt.tight_layout()
plt.show()


# In[36]:


# it seems there is more margin between listing and sale price for Adidas than for Nike.
# So, Adidas products would be having more discount as compare to Nike,Let's see.


# #### Discount by Brands

# In[37]:


plt.figure(figsize=(6, 3))

# Plot Discount % for Adidas and Nike on the same axis
sns.histplot(df1[df1["Merged_var_brand"] == "Adidas"]["Discount % age"], kde=False, label='Adidas')
sns.histplot(df1[df1["Merged_var_brand"] == "Nike"]["Discount % age"], kde=False, label='Nike')

# Add labels, title, and legend
plt.xticks(rotation=45)
plt.title('Discount by Brands')
plt.xlabel('Discount %')
plt.ylabel('Frequency')
plt.legend()

plt.show()


# # Granular Analysis

# #### Correlation between Rating and other variables

# In[38]:


# Selecting relevant columns for correlation

df_corr = df1[['Rating', 'Reviews', 'Listing Price', 'Sale Price', 'Brand', 'Discount % age']]


df_corr = pd.get_dummies(df_corr, columns=['Brand'])


df_corr.head()




# In[39]:


plt.figure(figsize=(10, 8))
sns.heatmap(df_corr.corr(), annot=True, cmap='Reds', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# ### General Observations
# 
# **Strong Positive Correlation**:
# Sale Price and Listing Price (0.89): This high correlation indicates that as the listing price increases, the sale price tends to increase as well, which is expected. It suggests that the pricing strategy might be consistent across products.
# 
# **Sale Price for Nike products is relatively higher than that of Adidas products.**
# 
# **Rating and Reviews (0.14)**: Although the correlation is positive, it is relatively weak. This could imply that higher-rated products do not necessarily receive significantly more reviews.
# 
# 
# **Negative Correlation**:
# Sale Price and Discount % age (-0.59): This indicates that as the discount percentage increases, the sale price tends to decrease. This is logical, as higher discounts typically result in lower sale prices.
# 
# **Brand_Nike and Rating (-0.18)**: This negative correlation suggests that products branded as Nike tend to have lower ratings in comparison to Adidas products.
# 
# **Brand-Specific Observations**:
# The correlations involving Brand_Adidas CORE / NEO, Brand_Adidas ORIGINALS, and Brand_Adidas SPORT PERFORMANCE show varying relationships with the other metrics, indicating that different Adidas lines might perform differently in terms of sales, pricing, and customer feedback.
# 
# **The correlation of Brand_Nike with reviews and other metrics is notably negative, highlighting that Nike products might not be rated as favorably as Adidas products in this dataset.**

# In[40]:


avg_metrics = Adidas_prod.groupby('Brand')[['Rating', 'Reviews', 'Listing Price', 'Sale Price']].mean().reset_index()


plt.figure(figsize=(15, 10))

# Plotting Average Rating
plt.subplot(2, 2, 1)
sns.barplot(data=avg_metrics, x='Brand', y='Rating', palette='Blues')
plt.title('Average Rating per Adidas Variant')
plt.xticks(rotation=45)

# Plotting Average no. of  Review
plt.subplot(2, 2, 2)
sns.barplot(data=avg_metrics, x='Brand', y='Reviews', palette='Greens')
plt.title('Average no. of Reviews per Adidas Variant')
plt.xticks(rotation=45)

# Plotting Average Listing Price
plt.subplot(2, 2, 3)
sns.barplot(data=avg_metrics, x='Brand', y='Listing Price', palette='Reds')
plt.title('Average Listing Price per Adidas Variant')
plt.xticks(rotation=45)

# Plotting Average Sale Price
plt.subplot(2, 2, 4)
sns.barplot(data=avg_metrics, x='Brand', y='Sale Price', palette='Purples')
plt.title('Average Sale Price per Adidas Variant')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# In[ ]:





# # Outlier Detection

# In[41]:


from scipy.stats import zscore

df2=df1[['Listing Price', 'Sale Price','Discount % age','Rating','Reviews']]
# Calculating Z-scores for each feature
z_scores = np.abs(zscore(df2))

# Defining a threshold for considering something as an outlier (e.g., Z-score > 3)
threshold = 3
outliers = np.where(z_scores > threshold)
outliers


# In[42]:


# Removing outliers
df2_ = df2[(z_scores < threshold).all(axis=1)]   #df2_ is dataframe after removing outliers
print(f"dataframe shape before outlier removal {df2.shape}\n" 
      f"dataframe shape after outlier removal {df2_.shape}")


# In[43]:


df2_


# # Scaling

# In[44]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df2_scaled = scaler.fit_transform(df2_)


# In[45]:


df2_scaled_feature=df2_scaled[:, [0, 1]]   #feature selection


# In[46]:


df2_scaled_feature


# #  K-Means Clustering

# In[47]:


df2_array = df2_.values
df2_array 


# In[48]:


from sklearn.cluster import KMeans

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    kmeans.fit(df2_array)
    inertia.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()


# In[49]:


#  3 clusters is the ideal choice since it is where the elbow occurs. 
# This point suggests that adding more clusters beyond this number does not substantially decrease inertia.


# In[50]:


# Convert df2_ to a NumPy array (if it's not already) for easier indexing
# kmeans for raw data

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=50)
clusters = pd.Series(kmeans.fit_predict(df2_array))


# Plot clusters with specific colors for each cluster
plt.scatter(df2_array[clusters == 0, 0], df2_array[clusters == 0, 1], s=30, c='red', label='Cluster 0')
plt.scatter(df2_array[clusters == 1, 0], df2_array[clusters== 1, 1], s=30, c='blue', label='Cluster 1')
plt.scatter(df2_array[clusters == 2, 0], df2_array[clusters== 2, 1], s=30, c='green', label='Cluster 2')

# Plot cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', marker='X', label='Centroids')

# Label the plot
plt.title('Clusters of shoes')
plt.xlabel('Listing price')
plt.ylabel('Sale price')
plt.legend()
plt.show()


# In[ ]:





# In[51]:


clusters_s = pd.Series(kmeans.fit_predict(df2_scaled_feature))  #scaled data and no.of clusters=3

plt.scatter(df2_scaled_feature[clusters_s  == 0, 0],df2_scaled_feature[clusters_s  == 0, 1], s=30, c='red', label='Cluster 0')
plt.scatter(df2_scaled_feature[clusters_s  == 1, 0],df2_scaled_feature[clusters_s == 1, 1], s=30, c='blue', label='Cluster 1')
plt.scatter(df2_scaled_feature[clusters_s == 2, 0],df2_scaled_feature[clusters_s == 2, 1], s=30, c='green', label='Cluster 2')

    

# Plotting the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', marker='X', label='Centroids')


# Adding titles and labels
plt.title('Clusters of shoes')
plt.xlabel('Listing price')
plt.ylabel('Sale price')
plt.legend()


# Show the plot
plt.show()


# In[52]:


kmeans_2 = KMeans(n_clusters=2,init='k-means++', max_iter=50)   # for scaled data & no.of clusters=2
clusters_2 = pd.Series(kmeans_2.fit_predict(df2_scaled_feature))

plt.scatter(df2_scaled_feature[clusters_2  == 0, 0],df2_scaled_feature[clusters_2  == 0, 1], s=30, c='red', label='Cluster 0')
plt.scatter(df2_scaled_feature[clusters_2  == 1, 0],df2_scaled_feature[clusters_2 == 1, 1], s=30, c='blue', label='Cluster 1')
plt.scatter(df2_scaled_feature[clusters_2 == 2, 0],df2_scaled_feature[clusters_2 == 2, 1], s=30, c='green', label='Cluster 2')

    

# Plotting the centroids
plt.scatter(kmeans_2.cluster_centers_[:, 0], kmeans_2.cluster_centers_[:, 1], s=100, c='black', marker='X', label='Centroids')


# Adding titles and labels
plt.title('Clusters of shoes')
plt.xlabel('Listing price')
plt.ylabel('Sale price')
plt.legend()


# Show the plot
plt.show()


# In[53]:


#pca data
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
df2_pca = pca.fit_transform(df2_scaled)


clusters_pca = pd.Series(kmeans.fit_predict(df2_pca))

plt.scatter(df2_pca[clusters_pca == 0, 0],df2_pca[clusters_pca == 0, 1], s=30, c='red', label='Cluster 0')
plt.scatter(df2_pca[clusters_pca == 1, 0],df2_pca[clusters_pca == 1, 1], s=30, c='blue', label='Cluster 1')
plt.scatter(df2_pca[clusters_pca == 2, 0],df2_pca[clusters_pca == 2, 1], s=30, c='green', label='Cluster 2')

# Plotting the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', marker='X', label='Centroids')


# Adding titles and labels
plt.title('Clusters of shoes')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Adding legend
plt.legend()

# Show the plot
plt.show()


# In[54]:


from sklearn.metrics import silhouette_score

silhouette_score(df2_array, clusters)   # for raw data


# In[55]:


silhouette_score(df2_scaled_feature, clusters_s)   # for scaled data & no.of clusters =3


# In[56]:


silhouette_score(df2_scaled_feature, clusters_2)   #for scaled data & no.of clusters =2


# In[57]:


silhouette_score(df2_pca, clusters_pca)    # for pca


# In[58]:


from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

# Function to plot silhouette scores
def plot_silhouette(data, clusters, n_clusters):
    # Calculate average silhouette score
    silhouette_avg = silhouette_score(data, clusters)
    sample_silhouette_values = silhouette_samples(data, clusters)

    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(8, 6)

    # Silhouette plot for each cluster
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate silhouette scores for samples in cluster i
        ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label clusters
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10  # Space between clusters

    ax1.set_title("Silhouette Plot for Clusters")
    ax1.set_xlabel("Silhouette Coefficient Values")
    ax1.set_ylabel("Cluster Label")

    # Plotting average silhouette score as a vertical line
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear y-axis labels/ticks
    ax1.set_xlim([-0.1, 1])
    plt.show()

# Plotting silhouette scores for different configurations

# For raw data clusters
plot_silhouette(df2_array, clusters, n_clusters=3)

# For clusters on scaled data (3 clusters)
plot_silhouette(df2_scaled_feature, clusters_s, n_clusters=3)

# For clusters on PCA data
plot_silhouette(df2_pca, clusters_pca, n_clusters=3)


# The PCA-transformed data shows a smoother silhouette plot, indicating potentially better-defined clusters. However, the lower average silhouette score suggests that some data points might be misclassified (as we see the cluster points with negative silhoutte score for pca data, are the most in number) or poorly assigned. This could be due to overfitting to noise or incorrect dimensionality reduction. For further analysis, consider visualizing the data in reduced dimensions, experimenting with different clustering algorithms, and evaluating the results using additional metrics

# ## Cluster Analysis

# In[59]:


import pandas as pd

# Assuming 'cluster_labels' is a Series containing cluster labels for each data point
df1['cluster'] = clusters_s

df1 = df1.dropna(subset=['cluster'])

# Group by cluster and analyze
for cluster_num in df1['cluster'].unique():
    cluster_data = df1[df1['cluster'] == cluster_num]
    
    # Brand distribution
    brand_counts = cluster_data['Brand'].value_counts()
    
    # Popular products
    product_counts = cluster_data['Product Name'].value_counts()
    
    print(f"Cluster {cluster_num}")
    print("-" * 20)
    print("Brand Distribution:")
    print(brand_counts.to_string())
    print("\nPopular Products:")
    print(product_counts.head(5).to_string())
    print("\n")


# In[60]:


# Visualization and descriptive statistics for each cluster

for cluster_num in df1['cluster'].unique():
    cluster_data = df1[df1['cluster'] == cluster_num]

    # Descriptive Statistics
    print(f"Cluster {cluster_num}")
    print(cluster_data[['Listing Price', 'Sale Price']].describe())

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.histplot(cluster_data['Listing Price'], kde=True)
    plt.title(f"Listing Price Distribution for Cluster {cluster_num}")
    plt.xlabel("Listing Price")
    plt.ylabel("Frequency")
    plt.show()

    sns.boxplot(x='Brand', y='Listing Price', data=cluster_data)
    plt.title(f"Box Plot of Listing Price by Brand for Cluster {cluster_num}")
    plt.xlabel("Brand")
    plt.ylabel("Listing Price")
    plt.xticks(rotation=45)
    plt.show()

    sns.scatterplot(x='Listing Price', y='Sale Price', hue='Brand', data=cluster_data)
    plt.title(f"Scatter Plot of Listing Price vs. Sale Price for Cluster {cluster_num}")
    plt.xlabel("Listing Price")
    plt.ylabel("Sale Price")
    plt.show()


# In[61]:


df1


# Availability column was not used in this study: however for future studies it may be useful.

# In[62]:


# Saving the cleaned DataFrame to a CSV file
df1.to_csv('cleaned_AdidasVsNike.csv', index=False)

