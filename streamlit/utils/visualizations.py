"""
Visualization functions for Streamlit app components.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import sys
import os

# Add the project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src import config

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

def plot_rfm_distribution(rfm_df):
    """
    Create distribution plots for RFM scores.
    
    Args:
        rfm_df: DataFrame with RFM scores
        
    Returns:
        fig: Matplotlib figure
    """
    rfm_score_cols = ["Recency_Score", "Frequency_Score", "Monetary_Score"]
    rfm_cols = ["Recency", "Frequency", "Monetary"]
    
    if not all(col in rfm_df.columns for col in rfm_score_cols):
        rfm_score_cols = ["R_Score", "F_Score", "M_Score"]
    
    if not all(col in rfm_df.columns for col in rfm_score_cols):
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, col in enumerate(rfm_score_cols):
        sns.histplot(rfm_df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {rfm_cols[i]} Score')
        axes[i].set_xlabel(f'{rfm_cols[i]} Score')
        axes[i].set_ylabel('Count')
    
    plt.tight_layout()
    return fig

def plot_rfm_heatmap(rfm_df):
    """
    Create a heatmap showing relationship between RFM scores.
    
    Args:
        rfm_df: DataFrame with RFM scores
        
    Returns:
        fig: Matplotlib figure
    """
    # Check for different column naming patterns
    if "R_Score" in rfm_df.columns:
        r_col, f_col, m_col = "R_Score", "F_Score", "M_Score"
    else:
        r_col, f_col, m_col = "Recency_Score", "Frequency_Score", "Monetary_Score"
    
    # Ensure columns exist
    if not all(col in rfm_df.columns for col in [r_col, f_col, m_col]):
        return None
    
    # Create pivot table for R-F relationship
    rf_heatmap = pd.pivot_table(rfm_df, values=m_col, 
                               index=r_col, columns=f_col, 
                               aggfunc='count')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(rf_heatmap, annot=True, cmap="YlGnBu", fmt="d", ax=ax)
    ax.set_title('Frequency vs Recency Heatmap (count of customers)')
    ax.set_xlabel('Frequency Score')
    ax.set_ylabel('Recency Score')
    
    plt.tight_layout()
    return fig

def plot_cluster_radar(segmentation_df, cluster_col='Cluster_Name', columns=None):
    """
    Create radar charts for each cluster showing average RFM values.
    
    Args:
        segmentation_df: DataFrame with cluster assignments and RFM values
        cluster_col: Column name for cluster assignments
        columns: List of columns to include in the radar chart (default: RFM values)
        
    Returns:
        fig: Matplotlib figure
    """
    if columns is None:
        # Try to find RFM columns with different naming patterns
        if "Recency" in segmentation_df.columns:
            columns = ["Recency", "Frequency", "Monetary"]
        elif "R_Value" in segmentation_df.columns:
            columns = ["R_Value", "F_Value", "M_Value"]
        else:
            columns = segmentation_df.select_dtypes(include=[np.number]).columns[:3]
    
    # Check if we have the necessary columns
    if not all(col in segmentation_df.columns for col in columns):
        return None
    
    if cluster_col not in segmentation_df.columns:
        return None
    
    # Get unique clusters
    clusters = segmentation_df[cluster_col].unique()
    n_clusters = len(clusters)
    
    if n_clusters > 6:  # Too many clusters for one radar chart
        # Just use the first 6 clusters
        clusters = clusters[:6]
        n_clusters = 6
    
    # Compute mean values for each cluster
    cluster_means = segmentation_df.groupby(cluster_col)[columns].mean()
    
    # Normalize the values for radar chart
    cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    
    # Create radar chart
    fig = plt.figure(figsize=(12, 8))
    
    # Angle for each attribute
    angles = np.linspace(0, 2*np.pi, len(columns), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    # Plot for each cluster
    ax = fig.add_subplot(111, polar=True)
    
    for i, cluster in enumerate(clusters):
        values = cluster_means_norm.loc[cluster].values.tolist()
        values += values[:1]  # Close the circle
        
        color = plt.cm.tab10(i / n_clusters)
        ax.plot(angles, values, 'o-', linewidth=2, label=f'{cluster}', color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Set labels and title
    ax.set_thetagrids(np.degrees(angles[:-1]), columns)
    ax.set_title('Cluster Profiles: Normalized RFM Values')
    ax.set_ylim(0, 1.05)  # Normalize between 0-1 with small margin
    ax.grid(True)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    plt.tight_layout()
    return fig

def plot_elasticity_distribution(elasticity_df, elasticity_col='Price_Elasticity_SKU_A', category_col=None):
    """
    Plot distribution of elasticity values, optionally by category.
    
    Args:
        elasticity_df: DataFrame with elasticity values
        elasticity_col: Column name for elasticity values
        category_col: Column name for categories (optional)
        
    Returns:
        fig: Matplotlib figure
    """
    # Filter to just own-price elasticity data
    own_elasticity_df = elasticity_df[elasticity_df['SKU_B'].isna()].copy()
    
    if elasticity_col not in own_elasticity_df.columns:
        return None
    
    if category_col and category_col in own_elasticity_df.columns:
        # Plot by category
        categories = own_elasticity_df[category_col].unique()
        n_categories = len(categories)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, category in enumerate(categories):
            category_data = own_elasticity_df[own_elasticity_df[category_col] == category]
            sns.kdeplot(category_data[elasticity_col], label=category, ax=ax)
        
        # Add a vertical line at x=0
        ax.axvline(x=0, color='red', linestyle='--')
        
        ax.set_title('Distribution of Own-Price Elasticity by Category')
        ax.set_xlabel('Elasticity Value')
        ax.set_ylabel('Density')
        ax.legend()
        
    else:
        # Plot overall distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.histplot(own_elasticity_df[elasticity_col], kde=True, ax=ax)
        
        # Add a vertical line at x=0
        ax.axvline(x=0, color='red', linestyle='--')
        
        ax.set_title('Distribution of Own-Price Elasticity')
        ax.set_xlabel('Elasticity Value')
        ax.set_ylabel('Count')
    
    plt.tight_layout()
    return fig

def plot_cross_elasticity_heatmap(elasticity_df):
    """
    Create a heatmap of cross-price elasticities between SKUs.
    
    Args:
        elasticity_df: DataFrame with cross-price elasticity values
        
    Returns:
        fig: Matplotlib figure
    """
    # Filter to just cross-price elasticity data
    cross_elasticity_df = elasticity_df[elasticity_df['SKU_B'].notna()].copy()
    
    if cross_elasticity_df.empty:
        return None
    
    if 'Price_Elasticity_SKU_A' not in cross_elasticity_df.columns:
        return None
    
    # Create pivot table for cross-elasticity heatmap
    pivot_df = cross_elasticity_df.pivot_table(
        index='SKU_A_Desc', 
        columns='SKU_B_Desc', 
        values='Price_Elasticity_SKU_A'
    )
    
    # If too many products, limit to top N with highest absolute elasticity
    if pivot_df.shape[0] > 15 or pivot_df.shape[1] > 15:
        # Calculate average absolute elasticity for each product
        absolute_elasticity = cross_elasticity_df.groupby('SKU_A_Desc')['Price_Elasticity_SKU_A'].apply(
            lambda x: abs(x).mean()
        ).sort_values(ascending=False)
        
        top_products = absolute_elasticity.head(15).index.tolist()
        
        # Filter pivot table to top products
        pivot_df = pivot_df.loc[
            pivot_df.index.isin(top_products), 
            pivot_df.columns.isin(top_products)
        ]
    
    # Create custom colormap: red for negative (complements), blue for positive (substitutes)
    colors = [(0.8, 0.2, 0.2), (1, 1, 1), (0.2, 0.2, 0.8)]  # Red -> White -> Blue
    cmap = LinearSegmentedColormap.from_list('RWB', colors, N=100)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Find max absolute value for symmetric color scale
    abs_max = abs(pivot_df.values).max()
    
    sns.heatmap(
        pivot_df, 
        cmap=cmap, 
        center=0, 
        vmin=-abs_max,
        vmax=abs_max,
        annot=True,
        fmt='.1f',
        ax=ax,
        cbar_kws={'label': 'Cross-Price Elasticity'}
    )
    
    ax.set_title('Cross-Price Elasticities Between Products')
    ax.set_xlabel('Price Change Product')
    ax.set_ylabel('Quantity Change Product')
    
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig

def plot_optimization_results(optimization_df, limit=None):
    """
    Plot optimization results showing original vs optimized revenue by SKU.
    
    Args:
        optimization_df: DataFrame with optimization results
        limit: Number of top SKUs to show (by revenue change)
        
    Returns:
        fig: Matplotlib figure
    """
    if optimization_df is None or optimization_df.empty:
        return None
    
    required_cols = ['Description', 'Original_Revenue', 'Optimized_Revenue', 'Revenue_Change_Pct']
    if not all(col in optimization_df.columns for col in required_cols):
        return None
    
    # Sort by absolute revenue change percentage
    sorted_df = optimization_df.sort_values('Revenue_Change_Pct', ascending=False)
    
    if limit and len(sorted_df) > limit:
        sorted_df = sorted_df.head(limit)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    x = np.arange(len(sorted_df))
    width = 0.35
    
    # Plot original and optimized revenue as grouped bars
    ax.bar(x - width/2, sorted_df['Original_Revenue'], width, label='Original Revenue')
    ax.bar(x + width/2, sorted_df['Optimized_Revenue'], width, label='Optimized Revenue')
    
    # Add percentage change as text
    for i, row in enumerate(sorted_df.itertuples()):
        change_pct = row.Revenue_Change_Pct
        color = 'green' if change_pct > 0 else 'red'
        ax.text(i, max(row.Original_Revenue, row.Optimized_Revenue) + 5, 
                f"{change_pct:.1f}%", ha='center', color=color, fontweight='bold')
    
    ax.set_title('Original vs. Optimized Revenue by Product')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_df['Description'], rotation=90)
    ax.set_ylabel('Revenue')
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_price_change_impact(optimization_df):
    """
    Create a scatter plot showing the relationship between price changes and revenue changes.
    
    Args:
        optimization_df: DataFrame with optimization results
        
    Returns:
        fig: Matplotlib figure
    """
    if optimization_df is None or optimization_df.empty:
        return None
    
    required_cols = ['Price_Change_Pct', 'Revenue_Change_Pct', 'Original_Revenue']
    if not all(col in optimization_df.columns for col in required_cols):
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot with point size proportional to original revenue
    sizes = optimization_df['Original_Revenue'] / optimization_df['Original_Revenue'].max() * 300
    
    scatter = ax.scatter(
        optimization_df['Price_Change_Pct'],
        optimization_df['Revenue_Change_Pct'],
        s=sizes,
        c=optimization_df['Revenue_Change_Pct'],
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='k',
        linewidths=1
    )
    
    # Add quadrant lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    
    # Add labels for points
    for i, row in enumerate(optimization_df.itertuples()):
        ax.annotate(
            row.Description[:10] + "..." if len(row.Description) > 10 else row.Description,
            (row.Price_Change_Pct, row.Revenue_Change_Pct),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.8
        )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Revenue Change (%)')
    
    ax.set_title('Price Change vs. Revenue Change')
    ax.set_xlabel('Price Change (%)')
    ax.set_ylabel('Revenue Change (%)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig