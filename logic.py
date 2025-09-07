from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime
from datetime import timedelta 
import re
import gc
from sklearn.neighbors import NearestNeighbors
import warnings
from tqdm import tqdm 
from openai import OpenAI
import numpy as np
from gensim.models import Word2Vec # type: ignore
import json
import time
import nltk
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def classify_items(file_path):
    result_df = pd.read_excel(file_path)
    return result_df


def process_csv_and_config(df, config_dict):
    import re
    from datetime import datetime

    # Clean column names
    df.columns = [
        re.sub(r'[^a-zA-Z0-9]+', '_', col).strip('_').lower()
        for col in df.columns
    ]

    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['sales_price'] = pd.to_numeric(df['sales_price'], errors='coerce')
    df['est_unit_cost'] = pd.to_numeric(df['est_unit_cost'], errors='coerce')

    df = df.dropna(subset=['transaction_date', 'quantity', 'sales_price', 'est_unit_cost']).reset_index(drop=True)
    df = df[df['sales_price'] >= df['est_unit_cost']]
    df = df[df['transaction_date'] <= pd.Timestamp(datetime.now().date())]

    def cap_outliers(group):
        numeric_cols = group.select_dtypes(include=[float, int]).columns
        for col in numeric_cols:
            if 'id' not in col.lower():
                q1, q99 = group[col].quantile(0.01), group[col].quantile(0.99)
                group[col] = group[col].clip(lower=q1, upper=q99)
        return group

    df = df.groupby('item_name', group_keys=False).apply(cap_outliers).drop_duplicates().dropna()

    return df





def classify_products(transaction_df, config):
    
    merged_df = transaction_df.copy()
    merged_df = merged_df.dropna()
    merged_df['transaction_date'] = pd.to_datetime(merged_df['transaction_date'])
    merged_df['revenue'] = merged_df['quantity'] * merged_df['sales_price']
    merged_df['margin'] = merged_df['revenue'] - (merged_df['quantity'] * merged_df['est_unit_cost'])

    latest_date = merged_df['transaction_date'].max()

    # New or Old classification
    group = merged_df.groupby('item_name')
    first_tx_date = group['transaction_date'].min().reset_index()
    first_tx_date['New/Old'] = np.where(
        first_tx_date['transaction_date'] >= latest_date - pd.DateOffset(months=config["threshold_months"]),
        'New Launch', 'Old Launch'
    )

    # Top 80% Revenue
    revenue_by_item = group['revenue'].sum().reset_index().sort_values(by='revenue', ascending=False)
    revenue_by_item['cumsum'] = revenue_by_item['revenue'].cumsum()
    revenue_by_item['cum_pct'] = revenue_by_item['cumsum'] / revenue_by_item['revenue'].sum()
    revenue_by_item['Top 80%?'] = np.where(revenue_by_item['cum_pct'] <= (config["data_divide_percent"])/100, 'Yes', 'No')

    # YOY Calculation
    one_year_ago = latest_date - pd.DateOffset(years=1)
    two_years_ago = latest_date - pd.DateOffset(years=2)

    df_yoy = merged_df.copy()
    df_yoy['year_window'] = np.where(
        df_yoy['transaction_date'] > one_year_ago, 'current_year',
        np.where(df_yoy['transaction_date'] > two_years_ago, 'previous_year', 'older')
    )

    revenue_yoy = df_yoy[df_yoy['year_window'].isin(['current_year', 'previous_year'])]\
        .groupby(['item_name', 'year_window'])['revenue'].sum().unstack(fill_value=0).reset_index()
    revenue_yoy['YOY Revenue'] = np.where(
        revenue_yoy['current_year'] > revenue_yoy['previous_year'],
        'Increase', 'Decrease'
    )

    margin_yoy = df_yoy[df_yoy['year_window'].isin(['current_year', 'previous_year'])]\
        .groupby(['item_name', 'year_window'])['margin'].sum().unstack(fill_value=0).reset_index()
    margin_yoy['YOY Margin'] = np.where(
        margin_yoy['current_year'] > margin_yoy['previous_year'],
        'Increase', 'Decrease'
    )

    # Merge all features
    final_df = first_tx_date[['item_name', 'New/Old']]\
        .merge(revenue_by_item[['item_name', 'Top 80%?']], on='item_name', how='left')\
        .merge(revenue_yoy[['item_name', 'YOY Revenue']], on='item_name', how='left')\
        .merge(margin_yoy[['item_name', 'YOY Margin']], on='item_name', how='left')\
        .sort_values(by='item_name').reset_index(drop=True)

    # Add revenue and classification placeholder
    df = final_df.copy()
    item_revenue = merged_df.groupby('item_name')['revenue'].sum().reset_index()
    df = df.merge(item_revenue, on='item_name', how='left')
    df['class'] = 'not_applicable'

    # Old Bottom 20% Analysis
    mask_old = (df['New/Old'] == 'Old Launch') & (df['Top 80%?'] == 'No')
    df_old = df[mask_old].sort_values(by='revenue', ascending=False).copy()
    df_old['cumsum'] = df_old['revenue'].cumsum()
    total_old = df_old['revenue'].sum()
    df_old['cum_pct'] = df_old['cumsum'] / total_old
    old_top_items = set(df_old[df_old['cum_pct'] <= 0.2]['item_name'])

    df_old_bottom = df_old[df_old['cum_pct'] > 0.2].copy()
    df_old_bottom['cumsum2'] = df_old_bottom['revenue'].cumsum()
    total_old_bottom = df_old_bottom['revenue'].sum()
    df_old_bottom['cum_pct2'] = df_old_bottom['cumsum2'] / total_old_bottom
    old_bottom_top_items = set(df_old_bottom[df_old_bottom['cum_pct2'] <= 0.2]['item_name'])

    df.loc[df['item_name'].isin(old_top_items), 'class'] = 'old_bottom20_top20'
    df.loc[df['item_name'].isin(old_bottom_top_items), 'class'] = 'old_bottom20_bottom80_top20'
    df.loc[(mask_old) & (~df['item_name'].isin(old_top_items)) & (~df['item_name'].isin(old_bottom_top_items)),
           'class'] = 'old_bottom20_bottom80_bottom80'

    # New Bottom 20% Analysis
    mask_new = (df['New/Old'] == 'New Launch') & (df['Top 80%?'] == 'No')
    df_new = df[mask_new].sort_values(by='revenue', ascending=False).copy()
    df_new['cumsum'] = df_new['revenue'].cumsum()
    total_new = df_new['revenue'].sum()
    df_new['cum_pct'] = df_new['cumsum'] / total_new
    new_top_items = set(df_new[df_new['cum_pct'] <= 0.2]['item_name'])

    df_new_bottom = df_new[df_new['cum_pct'] > 0.2].copy()
    df_new_bottom['cumsum2'] = df_new_bottom['revenue'].cumsum()
    total_new_bottom = df_new_bottom['revenue'].sum()
    df_new_bottom['cum_pct2'] = df_new_bottom['cumsum2'] / total_new_bottom
    new_bottom_top_items = set(df_new_bottom[df_new_bottom['cum_pct2'] <= 0.2]['item_name'])

    df.loc[df['item_name'].isin(new_top_items), 'class'] = 'new_bottom20_top20'
    df.loc[df['item_name'].isin(new_bottom_top_items), 'class'] = 'new_bottom20_bottom80_top20'
    df.loc[(mask_new) & (~df['item_name'].isin(new_top_items)) & (~df['item_name'].isin(new_bottom_top_items)),
           'class'] = 'new_bottom20_bottom80_bottom80'

    df = df.drop(columns=['revenue', 'cumsum', 'cum_pct'], errors='ignore')
    # Decision Label Logic
    conditions = [
        (df["New/Old"] == "Old Launch") & (df["Top 80%?"] == "Yes") & (df["YOY Revenue"] == "Increase") & (df["YOY Margin"] == "Increase"),
        (df["New/Old"] == "Old Launch") & (df["Top 80%?"] == "Yes"),
        (df["New/Old"] == "New Launch") & (df["Top 80%?"] == "Yes") & (df["YOY Margin"] == "Increase"),
        (df["New/Old"] == "New Launch") & (df["Top 80%?"] == "Yes"),
        (df["New/Old"] == "Old Launch") & (df["class"] == "old_bottom20_top20") & (df["YOY Revenue"] == "Increase") & (df["YOY Margin"] == "Increase"),
        (df["New/Old"] == "Old Launch") & (df["class"] == "old_bottom20_top20") & (df["YOY Revenue"] == "Increase"),
        (df["New/Old"] == "Old Launch") & (df["class"] == "old_bottom20_bottom80_top20") & (df["YOY Revenue"] == "Decrease"),
        (df["New/Old"] == "Old Launch") & (df["class"] == "old_bottom20_bottom80_bottom80") & (df["YOY Revenue"] == "Decrease") & (df["YOY Margin"] == "Increase"),
        (df["New/Old"] == "Old Launch") & (df["class"] == "old_bottom20_bottom80_bottom80") & (df["YOY Revenue"] == "Decrease") & (df["YOY Margin"] == "Decrease"),
        (df["New/Old"] == "New Launch") & (df["class"] == "new_bottom20_top20") & (df["YOY Margin"] == "Increase"),
        (df["New/Old"] == "New Launch") & (df["class"] == "new_bottom20_top20") & (df["YOY Margin"] == "Decrease"),
        (df["New/Old"] == "New Launch") & (df["class"] == "new_bottom20_bottom80_top20"),
        (df["New/Old"] == "New Launch") & (df["class"] == "new_bottom20_bottom80_bottom80") & (df["YOY Margin"] == "Increase"),
        (df["New/Old"] == "New Launch") & (df["class"] == "new_bottom20_bottom80_bottom80") & (df["YOY Margin"] == "Decrease"),
        (df["New/Old"] == "Old Launch") & (df["class"] == "old_bottom20_bottom80_bottom80") & ((df["YOY Revenue"] == "Increase") | (df["YOY Margin"] == "Increase")),
        (df["New/Old"] == "Old Launch") & (df["class"] == "old_bottom20_top20"),
        (df["New/Old"] == "Old Launch") & (df["class"] == "old_bottom20_bottom80_top20")
    ]

    choices = [
        "Invest", "Sustain", "Invest", "Sustain", "Sustain", "Watch",
        "Watch", "Watch", "Sunset", "Sustain", "Watch", "Watch",
        "Watch", "Sunset", "Watch", "Watch", "Watch"
    ]

    df["Label"] = np.select(conditions, choices, default="Not Sure")

    mandatory_items = config.get("mandatory_product", [])
    mandatory_list = []

    if isinstance(mandatory_items, str):
        mandatory_list = [item.strip() for item in mandatory_items.split(",") if item.strip()]
    elif isinstance(mandatory_items, list):
        mandatory_list = [str(item).strip() for item in mandatory_items if str(item).strip()]

    if mandatory_list and 'item_name' in df.columns:
        df['item_name'] = df['item_name'].astype(str).str.strip()
        df.loc[df['item_name'].isin(mandatory_list), 'Label'] = 'Invest'


    return df


def identify_label(df):
    # Return the split
    product_with_label = df[df["Label"] != "Not Sure"]
    product_without_label = df[df["Label"] == "Not Sure"]
    return product_with_label, product_without_label

def item_with_parent(df, result):
    return pd.merge(df, result, on='item_name', how='left')


def tokenize_item_names(df, col_name='item_name'):
   
    df = df.copy()  
    df['item_tokens'] = df[col_name].apply(lambda x: word_tokenize(str(x).lower()))
    return df

def train_word2vec(df, token_col='item_tokens', vector_size=50, window=3, min_count=1, workers=4):
    
    corpus = df[token_col].tolist()
    model = Word2Vec(sentences=corpus, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model

def get_w2v_vector(tokens, model, vector_size):
    """
    Get average Word2Vec vector for a list of tokens.
    """
    valid_vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not valid_vectors:
        return np.zeros(vector_size)
    return np.mean(valid_vectors, axis=0)

def add_item_vectors(df, model, token_col='item_tokens', vector_col='item_vector', vector_size=50):
  
    df = df.copy()  
    df[vector_col] = df[token_col].apply(lambda tokens: get_w2v_vector(tokens, model, vector_size))
    return df

def generate_item_vectors(df, col_name='item_name', vector_size=50, window=3, min_count=1, workers=4):
    
    df = df.copy()

    df['item_tokens'] = df[col_name].apply(lambda x: word_tokenize(str(x).lower()))

    corpus = df['item_tokens'].tolist()
    model = Word2Vec(sentences=corpus, vector_size=vector_size, window=window, min_count=min_count, workers=workers)

    def get_vector(tokens):
        valid_vectors = [model.wv[token] for token in tokens if token in model.wv]
        if not valid_vectors:
            return np.zeros(vector_size)
        return np.mean(valid_vectors, axis=0).tolist()  

    df['item_vector'] = df['item_tokens'].apply(get_vector)

    return df


def find_similar_items(new_df, old_df, vector_col='item_vector', n_neighbors=5):
    similar_items_data = []

    old_df = old_df.copy()
    new_df = new_df.copy()
    old_df[vector_col] = old_df[vector_col].apply(lambda x: np.array(x))
    new_df[vector_col] = new_df[vector_col].apply(lambda x: np.array(x))

    for _, new_row in new_df.iterrows():
        new_item_name = new_row['item_name']
        parent_class = new_row['parent_class']
        new_vector = new_row[vector_col]

        filtered_old = old_df[old_df['parent_class'] == parent_class]

        if len(filtered_old) == 0:
            continue  # No match in parent class

        X = np.vstack(filtered_old[vector_col].values)
        knn = NearestNeighbors(n_neighbors=min(n_neighbors, len(filtered_old)), metric='cosine')
        knn.fit(X)

        distances, indices = knn.kneighbors([new_vector])

        for dist, idx in zip(distances[0], indices[0]):
            matched_row = filtered_old.iloc[idx]
            similar_items_data.append({
                'item_name': new_item_name,
                'similar_item': matched_row['item_name'],
                'similar_item_parent': matched_row['parent_class'],
                'similar_item_label': matched_row['Label']
            })
    similar_df = pd.DataFrame(similar_items_data)
    remaining_item_df = new_df[~new_df["item_name"].isin(similar_df["item_name"])][["item_name", "parent_class"]].reset_index(drop=True)


    return similar_df, remaining_item_df


def assign_max_priority_label(similar_df):
    from collections import Counter
    priority_order = {'Invest': 4, 'Sustain': 3, 'Watch': 2, 'Sunset': 1}

    def resolve_label_and_parent(group):
        # Count labels
        label_counts = Counter(group['similar_item_label'])
        max_count = max(label_counts.values())

        # Get all labels with the highest count
        candidates = [label for label, count in label_counts.items() if count == max_count]

        # Select highest priority label from candidates
        final_label = max(candidates, key=lambda x: priority_order[x])

        # Since all parent values are same for a group, just take first
        parent = group['similar_item_parent'].iloc[0]

        return pd.Series({
            'label': final_label,
            'similar_item_parent': parent
        })

    # Group and apply
    result_df = similar_df.groupby('item_name').apply(resolve_label_and_parent).reset_index()

    return result_df


def prepared_final_files(final_labels_df, product_with_label_vec, remaining_df):
    final_labels_df_renamed = final_labels_df.rename(columns={
    "similar_item_parent": "parent_class",
    "label": "Label"
    })

     # Step 1: Combine product_with_label_vec and final_labels_df
    all_df_with_label = pd.concat([
        product_with_label_vec[["item_name", "parent_class", "Label"]],
        final_labels_df_renamed[["item_name", "parent_class", "Label"]]
    ], axis=0)

    # Step 2: Prepare remaining_df with same column names
    remaining_df_cleaned = remaining_df.copy()
    remaining_df_cleaned["Label"] = "Not Sure"
    remaining_df_cleaned = remaining_df_cleaned[["item_name", "parent_class", "Label"]]

    # Step 3: Concatenate all into final DataFrame
    final_combined_df = pd.concat([all_df_with_label, remaining_df_cleaned], axis=0).reset_index(drop=True)
    non_null_df = final_combined_df[final_combined_df['parent_class'].notna()]
    null_df = final_combined_df[final_combined_df['parent_class'].isna()]
    final_combined_df = pd.concat([non_null_df, null_df], ignore_index=True)


    return final_combined_df


# def summarize_labels_by_parent_class(df):
#     # Group and count
#     grouped = df.groupby(['parent_class', 'Label']).size().unstack(fill_value=0)

#     # Ensure all expected labels are present
#     for col in ['Invest', 'Sustain', 'Watch', 'Sunset', 'Not Sure']:
#         if col not in grouped.columns:
#             grouped[col] = 0

#     # Prepare label count summary
#     parent_classification_df = grouped[['Invest', 'Sustain', 'Watch', 'Sunset', 'Not Sure']].reset_index()
#     parent_classification_df.columns.name = None

#     # Calculate weighted confidence
#     grouped_conf = grouped.copy()
#     numerator = (
#         grouped_conf['Invest'] * 1 +
#         grouped_conf['Sustain'] * 0.8 +
#         grouped_conf['Watch'] * 0.6 +
#         grouped_conf['Sunset'] * 0.2
#     )
#     denominator = (
#         grouped_conf['Invest'] +
#         grouped_conf['Sustain'] +
#         grouped_conf['Watch'] +
#         grouped_conf['Sunset']
#     )

#     grouped_conf['confidence'] = numerator / denominator.replace(0, float('nan'))  # avoid divide-by-zero warning
#     grouped_conf['confidence'] = grouped_conf['confidence'].fillna(0)  # replace NaN with 0

#     confidence_df = grouped_conf[['confidence']].reset_index()

#     return parent_classification_df, confidence_df

def summarize_labels_by_parent_class(df):
    # Group and count
    grouped = df.groupby(['parent_class', 'Label']).size().unstack(fill_value=0)

    # Ensure all expected labels are present
    for col in ['Invest', 'Sustain', 'Watch', 'Sunset', 'Not Sure']:
        if col not in grouped.columns:
            grouped[col] = 0

    # Calculate weighted confidence
    numerator = (
        grouped['Invest'] * 1 +
        grouped['Sustain'] * 0.8 +
        grouped['Watch'] * 0.6 +
        grouped['Sunset'] * 0.2
    )
    denominator = (
        grouped['Invest'] +
        grouped['Sustain'] +
        grouped['Watch'] +
        grouped['Sunset']
    )

    grouped['confidence'] = numerator / denominator.replace(0, float('nan'))
    grouped['confidence'] = grouped['confidence'].fillna(0)
    grouped['confidence'] = grouped['confidence'].round(2)

    # Final DataFrame with counts + confidence
    parent_classification_df = grouped[['Invest', 'Sustain', 'Watch', 'Sunset', 'Not Sure', 'confidence']].reset_index()
    parent_classification_df.columns.name = None

    return parent_classification_df



def prepare_ranked_output_by_margin(Final_result, transaction_df):
    # Step 1: Compute margin and total margin per item
    transaction_df['margin'] = transaction_df['sales_price'] - transaction_df['est_unit_cost']
    margin_df = (
        transaction_df
        .groupby('item_name', as_index=False)['margin']
        .sum()
    )

    # Step 2: Merge with Final_result on item_name
    merged_df = pd.merge(Final_result, margin_df, on='item_name', how='left')

    # Fill missing margins with 0
    merged_df['margin'] = merged_df['margin'].fillna(0)

    # Step 3: Define label sorting order
    label_order = {
        'invest': 0,
        'watch': 1,
        'sustain': 2,
        'sunset': 3,
        'not sure': 4
    }

    # Normalize Label column to lowercase for sorting
    merged_df['Label'] = merged_df['Label'].astype(str).str.lower()
    merged_df['label_order'] = merged_df['Label'].map(label_order).fillna(5)

    # Step 4: Sort by parent_class, label_order, then margin descending
    sorted_df = merged_df.sort_values(
        by=['parent_class', 'label_order', 'margin'],
        ascending=[True, True, False]
    )

    # Step 5: Final output with desired column order
    sorted_df = sorted_df[['parent_class', 'item_name', 'Label']]

    return sorted_df


# def split_final_result_by_label(Final_result):
#     # Normalize labels to lowercase
#     Final_result['Label'] = Final_result['Label'].astype(str).str.lower()

#     # Create individual DataFrames for each label
#     invest_df = Final_result[Final_result['Label'] == 'invest']
#     watch_df = Final_result[Final_result['Label'] == 'watch']
#     sustain_df = Final_result[Final_result['Label'] == 'sustain']
#     sunset_df = Final_result[Final_result['Label'] == 'sunset']

#     # Create count summary DataFrame
#     label_counts = {
#         'Invest': len(invest_df),
#         'Watch': len(watch_df),
#         'Sustain': len(sustain_df),
#         'Sunset': len(sunset_df)
#     }
#     summary_df = pd.DataFrame(list(label_counts.items()), columns=["Label", "Count"])
#     col_order = ["item_name", "parent_class", "Label"]
#     invest_df = invest_df[col_order]
#     watch_df = watch_df[col_order]
#     sustain_df = sustain_df[col_order]
#     sunset_df = sunset_df[col_order]
#     summary_df = summary_df[["Label", "Count"]]
#     return invest_df, watch_df, sustain_df, sunset_df, summary_df


def split_final_result_by_label(Final_result, transaction_df):
    import pandas as pd

    # Step 1: Normalize Label column
    Final_result['Label'] = Final_result['Label'].astype(str).str.lower()

    # Step 2: Calculate total margin per item_name
    transaction_df['total_margin'] = transaction_df['quantity'] * transaction_df['margin']
    margin_df = transaction_df.groupby('item_name', as_index=False)['total_margin'].sum()

    # Step 3: Merge margin into Final_result
    Final_result = Final_result.merge(margin_df, on='item_name', how='left')
    Final_result['total_margin'] = Final_result['total_margin'].fillna(0)

    # Step 4: Filter and sort by label and margin
    invest_df = Final_result[Final_result['Label'] == 'invest'].sort_values(by='total_margin', ascending=False)
    watch_df = Final_result[Final_result['Label'] == 'watch'].sort_values(by='total_margin', ascending=False)
    sustain_df = Final_result[Final_result['Label'] == 'sustain'].sort_values(by='total_margin', ascending=False)
    sunset_df = Final_result[Final_result['Label'] == 'sunset'].sort_values(by='total_margin', ascending=False)

    # Step 5: Summary count
    label_counts = {
        'Invest': len(invest_df),
        'Watch': len(watch_df),
        'Sustain': len(sustain_df),
        'Sunset': len(sunset_df)
    }
    summary_df = pd.DataFrame(list(label_counts.items()), columns=["Label", "Count"])
    total_count = summary_df["Count"].sum()
    summary_df["Percentage"] = (summary_df["Count"] / total_count * 100).round(2)
    # Step 6: Select and order columns
    col_order = ["item_name", "parent_class", "Label"]
    invest_df = invest_df[col_order]
    watch_df = watch_df[col_order]
    sustain_df = sustain_df[col_order]
    sunset_df = sunset_df[col_order]

    return invest_df, watch_df, sustain_df, sunset_df, summary_df




# def plot_label_distribution_pie(df):
#     df = df[df["Label"] != "not sure"]
#     label_counts = df['Label'].value_counts()
#     fig, ax = plt.subplots(figsize=(.5, .5))  # Slightly increased size for better fit
#     ax.pie(label_counts, labels=label_counts.index, autopct=lambda pct: f'{pct:.1f}%' if pct > 0 else '', startangle=90,
#            textprops={'fontsize': 3}, labeldistance=1.1)  # Reduced font size for labels to 6
#     # Set smaller font size for autopct values
#     plt.setp(ax.texts, fontsize=4)  # Reduced font size for all texts (labels and autopct) to 4
#     ax.axis('equal')  # Keep pie chart circular
#     return fig

import matplotlib.pyplot as plt

def plot_label_distribution_pie(df):
    df = df[df["Label"] != "not sure"]
    label_counts = df['Label'].value_counts()

    # Increase DPI for better scaling, reduce overall figure size
    fig, ax = plt.subplots(figsize=(1.5, 1.5), dpi=300)  # DPI is the key here

    ax.pie(
        label_counts,
        labels=label_counts.index,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 0 else '',
        startangle=90,
        textprops={'fontsize': 6},
        labeldistance=1.1
    )
    plt.setp(ax.texts, fontsize=5)
    ax.axis('equal')
    return fig

# def movement_of_sku_across_channel(df):
#     df["revenue"] = df["sales_price"] * df["quantity"]
#     df["margin"] = (df["sales_price"] - df["est_unit_cost"]) * df["quantity"]
#     grouped_df = df.groupby(["item_name", "channel"])[["revenue","margin"]].sum().reset_index()
#     margin_across_channel_df = grouped_df.pivot_table(index='item_name', columns='channel', values='margin', fill_value=0).reset_index()
#     revenue_across_channel_df = grouped_df.pivot_table(index='item_name', columns='channel', values='revenue', fill_value=0).reset_index()

#     return margin_across_channel_df, revenue_across_channel_df

def movement_of_sku_across_channel(df):
    # Calculate revenue and margin
    df["revenue"] = df["sales_price"] * df["quantity"]
    df["margin"] = (df["sales_price"] - df["est_unit_cost"]) * df["quantity"]

    # Group by item_name and channel
    grouped_df = df.groupby(["item_name", "channel"])[["revenue", "margin"]].sum().reset_index()

    # Pivot for margin
    margin_df = grouped_df.pivot_table(
        index="item_name", columns="channel", values="margin", fill_value=0
    )
    margin_df = margin_df.add_prefix("margin_").reset_index()

    # Pivot for revenue
    revenue_df = grouped_df.pivot_table(
        index="item_name", columns="channel", values="revenue", fill_value=0
    )
    revenue_df = revenue_df.add_prefix("revenue_").reset_index()

    # Merge both on item_name
    final_df = pd.merge(margin_df, revenue_df, on="item_name", how="outer")
    numeric_cols = final_df.select_dtypes(include=["float", "int"]).columns
    final_df[numeric_cols] = final_df[numeric_cols].round(1)

    return final_df
# def channel_wise_assortment(df, config):
#     df_web = df[df["channel"] == "web"]
#     df_bulk_order = df[df["channel"] == "Builk order"]
#     df_vending = df[df["channel"] == "Vending"]
#     df_store = df[df["channel"] == "Store"]
#     web_result = classify_products(df_web, config)
#     bulk_result = classify_products(df_bulk_order, config)
#     vending_result = classify_products(df_vending, config)
#     store_result = classify_products(df_store, config)
#     web_result = web_result[["item_name", "Label"]][web_result["Label"] != "Not Sure"]
#     bulk_result = bulk_result[["item_name", "Label"]][bulk_result["Label"] != "Not Sure"]
#     vending_result = vending_result[["item_name", "Label"]][vending_result["Label"] != "Not Sure"]
#     store_result = store_result[["item_name", "Label"]][store_result["Label"] != "Not Sure"]
#     print(web_result.shape, bulk_result.shape, vending_result.shape, store_result.shape)
#     return web_result, bulk_result, vending_result, store_result

def channel_wise_assortment(df, config):
    import pandas as pd

    # Split data by channel
    df_web = df[df["channel"] == "web"]
    df_bulk_order = df[df["channel"] == "Builk order"]
    df_vending = df[df["channel"] == "Vending"]
    df_store = df[df["channel"] == "Store"]

    # Classify products
    web_result = classify_products(df_web, config)
    bulk_result = classify_products(df_bulk_order, config)
    vending_result = classify_products(df_vending, config)
    store_result = classify_products(df_store, config)

    # Keep only item_name and Label, filter out "Not Sure"
    web_result = web_result[["item_name", "Label"]][web_result["Label"] != "Not Sure"].rename(columns={"Label": "Web Result"})
    bulk_result = bulk_result[["item_name", "Label"]][bulk_result["Label"] != "Not Sure"].rename(columns={"Label": "Bulk Result"})
    vending_result = vending_result[["item_name", "Label"]][vending_result["Label"] != "Not Sure"].rename(columns={"Label": "Vending Result"})
    store_result = store_result[["item_name", "Label"]][store_result["Label"] != "Not Sure"].rename(columns={"Label": "Store Result"})

    # Merge all results into one DataFrame
    final_df = pd.merge(vending_result, store_result, on="item_name", how="outer")
    final_df = pd.merge(final_df, web_result, on="item_name", how="outer")
    final_df = pd.merge(final_df, bulk_result, on="item_name", how="outer")

    # ✅ Ensure all required columns exist
    final_df = final_df.reindex(columns=["item_name", "Web Result", "Bulk Result", "Vending Result", "Store Result"])
    return final_df


# import pandas as pd
# import time
# import ollama # type: ignore
# # import requests  # ✅ Correct way




# # 1. Function to build the prompt
# def build_prompt(item_name):
#     return  f"""
# You are given a list of product `item_name` values.

# Your task is to assign one clear, specific, and standardized store-level category to each `item_name`.

# ---

# ### RULES:

# - Assign only one category per item. Use a single word only.
# - Use real store-level categories such as: "Shirt", "Hoodie", "Sweater", "Shoes", "Laptop", "Book", "Cushion", etc.
# - If you cannot confidently assign a store-level category, write the `item_name` exactly as is in lowercase without spaces (for example: "Cotton Bag With Zip" → "cottonbagwithzip").
# - Do NOT return generic or placeholder categories such as "Miscellaneous", "Other", "Unknown", or leave it empty/null.
# - Maintain identical categories for semantically identical items.
# - Do not use plural's in result.

# ---


# ### EXAMPLES:

#     # <Item_name>:                                                                                                                       <Category>
#     SAFETY GLASSES PYRAMEX OTS CLEAR E600                                                                                                 Safety Glass
#     ACRYLIC INK 1OZ B135  FLSHTNT                                                                                                         Ink
#     Champion U of A Hoodie - Charcoal Grey, Forest (Adult, Unisex) : Champion U of A Hoodie-FS-L                                           HOODIE
#     WHISTLE ECONO                                                                                                                         WHISTLE ECONO
#     WHISTLE LANYARD CORD ASTD                                                                                                              LANYARD
#     CHAMPION MENS FLEECE JOGGER PANT BEARS : CHAMPION MENS FLEECE JOGGER PANT BEARS L BK 70053BU                                          JOGGER
#     JANSPORT RIGHT PACK 46X33X21CM : JANSPORT RIGHT PACK 46X33X21CM RUSSET RED S19  BU JS00TYP704S                                        PACK
#     PENCIL CASE QUO VADIS : PENCIL CASE VIOLET BIG TRAPEZE CALFSKIN 22 X 7 CM                                                             PENCIL CASE
#     SEAP Crewneck Russell 698M : SEAP Crewneck Russell 698M Ash 2XL                                                                       CREWNECK RUSSELL
#     CO Chic Mug With Bamboo Lid : CO Chic Mug With Bamboo Lid-WHITE                                                                        MUG
#     CO Custom Bookmark with Tassels                                                                                                       BOOKMARK
#     CO Knit Scarf                                                                                                                         Scarf
#     DAKINE CAMPUS L 33L BURNISHED LILAC                                                                                                   DAKINE CAMPUS
#     CO King Edward Large Journal (5.5" x 8.25") : CO King Edward Large Journal (5.5" x 8.25")-BLACK                                       Journal
#     METALLIC TEXTURED GRIP STYLUS PEN - BLACK INK : METALLIC TEXTURED GRIP STYLUS PEN - BLACK INK-BLACK                                   STYLUS PEN
#     CO Nursing Crewneck Jerzees Nublend 562 : CO Nursing Crewneck Jerzees Nublend 562 Ash M NONE                                         Jerzees
#     CO Nursing Hoodie M&O Soft 3320 : CO Nursing Hoodie M&O Soft 3320 LNONE                                                              Hoodie
#     CO Nylon Polyester Vest : CO Nylon Polyester Vest-L-YELLOW                                                                             VEST
#     Augustana Tshirt w/1colour print : Augustana Tshirt w/1colour print L Red                                                              T-SHIRT
#     Augustana Transit Stainless Bottle 24 oz Black                                                                                         Bottle
#     CO Recycled Fashion Tote : CO Recycled Fashion Tote-WHITE                                                                               Bag
#     CO SLP Gildan Heavy Cotton Long Sleeve Shirt : CO SLP Gildan Heavy Cotton Long Sleeve Shirt Black L                                   Shirt 
#     CO Milltex Unisex Heavy Weight Quarter Zip 935 : CO Milltex Unisex Heavy Weight Quarter Zip 935 Forest 3XL                           QUARTER-ZIP
#     CO Custom Dress Socks - The Classic                                                                                                   Socks
#     PS MEDICINE CLASS OF 2028 Jerzees Nublend Full Zip 993 : PS MEDICINE CLASS OF 2028 Jerzees Nublend Full Zip 993 HGY M                 FULL-ZIP
#     CO Malibu Sunglasses : CO Malibu Sunglasses-Black                                                                                     GLASSES
#     CO Lionheart Varsity Jacket : CO Lionheart Varsity Jacket--                                                                             Jacket
#     CO UA NURSING koi Next Gen Good Vibe Jogger 6-Pockets Women Wine : CO UA NURSING koi Next Gen Good Vibe Jogger 6-Pockets Women Wine-M-   Jogger
#     CO LSA 2024 Hugo Boss Ribbon Pen                                                                                                           Pen
#     CO LSA 2024 High Sierra Freewheel Pro Wheeled Backpack                                                                                  Backpack
#     CO Custom Bookmark with Tassels                                                                                                          Bookmark
#     ---



# ### Format:
# <Category>

# ## Important - Category should be single value. Model should not return a paragraph . It should retrurn a single word

# {item_name}
# """

# # 2. Function to call Ollama model

   
  
# def call_ollama_model(prompt, model="mistral", retries=3, delay=.5):
#     for attempt in range(retries):
#         try:
#             response = ollama.chat(
#                 model=model,
#                 messages=[{"role": "user", "content": prompt}]
#             )
#             return response['message']['content'].strip().split("\n")[0]
#         except Exception as e:
#             print(f"⚠️ Attempt {attempt+1} failed: {e}")
#             if attempt < retries - 1:
#                 time.sleep(delay)
#             else:
#                 return " "


# # 3. Function to classify item names in a DataFrame
# def classify_items_with_ollama(df, sleep_sec=0.2):
#     results = []

#     unique_items = df['item_name'].dropna().unique()
#     print("unique_items = ",unique_items.shape)
#     count = 0
#     for item in unique_items:
#         count = count + 1
#         print("Count = ", count)
#         prompt = build_prompt(item)
#         category = call_ollama_model(prompt)

#         # Filter out long/invalid responses
#         if len(category.split()) <= 3:
#             results.append({'item_name': item, 'parent_class': category})
#         else:
#             results.append({'item_name': item, 'parent_class': ' '})

#         time.sleep(sleep_sec)  # avoid overloading model

#     result_df = pd.DataFrame(results)
#     return result_df


# def process_single_item(item, Final_result):

#     df1 = item[["item_name"]]
#     item = classify_items_with_ollama(df1)
    
#     product_with_label_vec = generate_item_vectors(item)
#     similar_df1, remaining_df1 = find_similar_items(product_with_label_vec, Final_result)
#     if(len(similar_df1) > 0):
#         final_labels_df1 = assign_max_priority_label(similar_df1)
#     print(final_labels_df1.head())
#     return final_labels_df1, remaining_df1 

# import ollama # type: ignore
# print("✅ Using Ollama from:", ollama.__file__)