import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="🛒 Market Basket Recommender")

st.title("🛒 Market Basket Product Recommender")

# Upload CSV
file = st.file_uploader("📂 Upload Online Retail CSV (converted from Excel)", type=['csv'])

@st.cache_data
def preprocess_data(file):
    df = pd.read_csv(file, encoding='ISO-8859-1')
    df = df.dropna(subset=['Description'])
    df = df[df['Country'] == 'United Kingdom']
    df = df[df['Quantity'] > 0]
    df = df[df['InvoiceNo'].astype(str).str.startswith('5')]
    basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
    basket = basket.gt(0)
    return basket

@st.cache_data
def generate_rules(basket):
    frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    return rules.sort_values(by='lift', ascending=False)

if file:
    st.success("✅ Preprocessing data...")
    basket = preprocess_data(file)

    st.success("🧠 Running Apriori algorithm...")
    rules = generate_rules(basket)

    if rules.empty:
        st.warning("⚠️ No strong rules found. Try reducing `min_support`.")
    else:
        product_list = sorted(set(item for items in rules['antecedents'] for item in items))
        selected_product = st.selectbox("🔍 Select a Product to See Recommendations", product_list)

        matching_rules = rules[rules['antecedents'].apply(lambda x: selected_product in x)]

        if not matching_rules.empty:
            st.markdown(f"### ✅ Customers who bought **{selected_product}** also frequently bought:")
            for _, row in matching_rules.iterrows():
                consequents = ', '.join(row['consequents'])
                st.markdown(f"- **{consequents}** (Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f})")
        else:
            st.info("ℹ️ No recommendations found for the selected product.")
else:
    st.info("📥 Please upload the dataset to continue.")
