# Woolf_MLDM_Pranjal-Boruah
# 
# HR Reward Preference Clustering Analysis
# 
In this project, we aim to help the HR department of a services company segment its employees based on their reward preferences. Employees rated various leisure and entertainment activities such as Sports, Religious activities, Nature excursions, Theatre coupons, Shopping coupons, and Picnic outings. By applying clustering analysis, we can identify distinct groups, allowing HR to tailor non-monetary rewards to each segment.

#
# 1. Data Generation & Overview
# 
# For this demonstration, we create a synthetic dataset with 50 employees. The ratings are on a scale of 1 to 10.

# Setting a seed for reproducibility

# Number of employees in our synthetic dataset

# Creating the synthetic dataset dictionary
data = 
{'Emp Id': np.arange (1, num_employees + 1),
'Sports': np.random.randint(1, 11, size=num_employees),
'Religious': np.random.randint(1, 11, size=num_employees),
'Nature': np.random.randint(1, 11, size=num_employees),
'Theatre': np.random.randint(1, 11, size=num_employees),
'Shopping': np.random.randint(1, 11, size=num_employees),
'Picnic': np.random.randint(1, 11, size=num_employees)}

# Converting the dictionary into a DataFrame
df = pd.DataFrame(data)
df.head()

# %% [markdown]
# ## 2. Exploratory Data Analysis (EDA)
# 
# We start by checking the summary statistics and visualizing the distributions and relationships among the rating features.

# %%
# Displaying descriptive statistics of the dataset
print("Descriptive Statistics:")
print(df.describe())

# %% [markdown]
# ### 2.1 Pairplot
# 
# A pairplot helps visualize the pairwise relationships and distributions among the features.

# %%
sns.pairplot(df.drop('Emp Id', axis=1))
plt.show()

# %% [markdown]
# ### 2.2 Correlation Matrix
# 
# The correlation matrix gives insights into the relationships between different activity ratings.

# %%
plt.figure(figsize=(8,6))
sns.heatmap(df.drop('Emp Id', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Reward Preferences")
plt.show()

# %% [markdown]
# ## 3. Data Preprocessing
# 
# Since clustering algorithms are sensitive to different scales, we standardize the data.

# %%
features = ['Sports', 'Religious', 'Nature', 'Theatre', 'Shopping', 'Picnic']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# %% [markdown]
# ## 4. Determining the Optimal Number of Clusters (Elbow Method)
# 
# We use the Elbow Method by plotting the inertia (sum of squared distances) for different numbers of clusters (k). The "elbow" point suggests a good choice for k.

# %%
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    kmeans_model.fit(df_scaled)
    inertia.append(kmeans_model.inertia_)

plt.figure(figsize=(8,6))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.xticks(k_range)
plt.show()

# %% [markdown]
# From the Elbow Plot, assume the optimal number of clusters is **3**. This can be cross-validated by further analysis if needed.

# %% [markdown]
# ## 5. Clustering Analysis using K-Means
# 
# With the optimal number of clusters determined, we apply the K-Means algorithm to segment the employees.

# %%
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)
df.head()

# %% [markdown]
# ## 6. Visualizing the Clusters with PCA
# 
# To better visualize the clusters, we reduce the data dimensions to 2 using PCA.

# %%
pca = PCA(n_components=2, random_state=42)
df_pca = pca.fit_transform(df_scaled)

plt.figure(figsize=(8,6))
scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['Cluster'], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Employee Clusters Visualized with PCA')
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.show()

# %% [markdown]
# ## 7. Cluster Interpretation & Business Recommendations
# 
# We now analyze each cluster by calculating the average ratings for each activity. This summary helps HR customize rewards for each segment.
# 
# **Example Interpretation:**
# - **Cluster 0:** High average ratings in *Shopping* and *Theatre* might suggest a preference for urban and entertainment-based rewards.
# - **Cluster 1:** High ratings in *Nature* and *Picnic* may lean towards outdoor and team-bonding activities.
# - **Cluster 2:** A balanced score across categories might indicate employees who are open to a mix of rewards.
# 
# Based on these observations, HR can design non-monetary rewards that correspond to each segment’s dominant preferences.

# %%
cluster_summary = df.groupby('Cluster')[features].mean().round(2)
print("Cluster Summary (Mean Ratings):")
print(cluster_summary)

# %% [markdown]
# ## 8. Conclusion
# 
# In this project, we:
# - Generated and explored a dataset of employee reward preferences.
# - Standardized and applied K-Means clustering to segment employees.
# - Visualized the clusters using PCA.
# - Analyzed the clusters to provide actionable recommendations for the HR department.
# 
# This analysis allows the company to offer tailored non-monetary incentives, ensuring employees receive rewards that truly match their interests.

# %%
# Optionally, save the final dataset with cluster labels to a CSV for reporting
df.to_csv("employee_reward_clusters.csv", index=False)
print("Final data with cluster labels has been saved as 'employee_reward_clusters.csv'.")
