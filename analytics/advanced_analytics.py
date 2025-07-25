import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

class AdvancedAnalytics:
    def __init__(self):
        self.figures_dir = "reports/figures"
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def descriptive_analytics(self, df, dataset_name):
        """Generate comprehensive descriptive analytics"""
        report = {
            'dataset_name': dataset_name,
            'summary_stats': df.describe(),
            'correlation_matrix': df.corr(),
            'data_types': df.dtypes,
            'missing_values': df.isnull().sum()
        }
        
        # Generate visualizations
        self.plot_correlation_heatmap(df, dataset_name)
        self.plot_distribution_analysis(df, dataset_name)
        
        return report
    
    def plot_correlation_heatmap(self, df, dataset_name):
        """Generate correlation heatmap"""
        plt.figure(figsize=(12, 10))
        correlation_matrix = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title(f'Correlation Heatmap - {dataset_name}')
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/correlation_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_distribution_analysis(self, df, dataset_name):
        """Plot distribution of key variables"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        n_cols = 3
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, col in enumerate(numeric_columns):
            if i < len(axes):
                axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(numeric_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/distributions_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def student_segmentation(self, df):
        """Perform student segmentation using clustering"""
        # Prepare features for clustering
        features = df.select_dtypes(include=[np.number]).fillna(0)
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Optimal number of clusters using elbow method
        inertias = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features_scaled)
            inertias.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.savefig(f'{self.figures_dir}/elbow_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Perform clustering with optimal k (let's use k=4)
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Add cluster labels to dataframe
        df_clustered = df.copy()
        df_clustered['cluster'] = clusters
        
        # PCA for visualization
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('Student Segmentation - PCA Visualization')
        plt.colorbar(scatter)
        plt.savefig(f'{self.figures_dir}/student_segmentation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return df_clustered, kmeans
    
    def anomaly_detection(self, df):
        """Detect anomalous student behavior"""
        features = df.select_dtypes(include=[np.number]).fillna(0)
        
        # Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(features)
        
        df_anomalies = df.copy()
        df_anomalies['is_anomaly'] = anomalies == -1
        
        # Visualize anomalies
        if len(features.columns) >= 2:
            plt.figure(figsize=(12, 8))
            normal_points = df_anomalies[df_anomalies['is_anomaly'] == False]
            anomaly_points = df_anomalies[df_anomalies['is_anomaly'] == True]
            
            plt.scatter(normal_points.iloc[:, 0], normal_points.iloc[:, 1], 
                       c='blue', alpha=0.6, label='Normal')
            plt.scatter(anomaly_points.iloc[:, 0], anomaly_points.iloc[:, 1], 
                       c='red', alpha=0.8, label='Anomaly')
            
            plt.xlabel(features.columns[0])
            plt.ylabel(features.columns[1])
            plt.title('Anomaly Detection in Student Data')
            plt.legend()
            plt.savefig(f'{self.figures_dir}/anomaly_detection.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return df_anomalies
    
    def predictive_insights(self, df, target_column):
        """Generate predictive insights and feature importance"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Prepare features and target
        features = df.select_dtypes(include=[np.number]).drop(columns=[target_column])
        target = df[target_column]
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance.head(10), y='feature', x='importance')
        plt.title('Top 10 Feature Importance for Prediction')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return feature_importance, rf_model

if __name__ == "__main__":
    analytics = AdvancedAnalytics()
    
    # Load sample data (replace with actual data loading)
    # df = pd.read_csv("data/processed/combined_dataset.csv")
    
    # Run analytics
    # report = analytics.descriptive_analytics(df, "student_performance")
    # df_clustered, kmeans_model = analytics.student_segmentation(df)
    # df_anomalies = analytics.anomaly_detection(df)
    # feature_importance, model = analytics.predictive_insights(df, "performance_label")
