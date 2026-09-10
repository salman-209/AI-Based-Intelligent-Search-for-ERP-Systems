import streamlit as st
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title='AI ERP Smart Search', page_icon='🔎', layout='wide')
st.title('🔎 AI-Based Intelligent Search for ERP')
st.write('Search ERP data using natural language')

TRAIN_PATH = r'C:\hadoop\My Data\Intern\Ecommerce Order Dataset\train'

def load_csv(name):
    path = os.path.join(TRAIN_PATH, name)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

products = load_csv('df_Products.csv')
payments = load_csv('df_Payments.csv')
orders = load_csv('df_Orders.csv')
customers = load_csv('df_Customers.csv')

if products.empty:
    st.error(f'df_Products.csv not found. Check: {TRAIN_PATH}')
    st.stop()

def find_col(df, names):
    if df.empty:
        return None
    exact = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in exact:
            return exact[n.lower()]
    for c in df.columns:
        for n in names:
            if n.lower() in str(c).strip().lower():
                return c
    return None

product_id_col = find_col(products, ['product_id','productid','product id','sku','sku_id'])
category_col = find_col(products, ['category','product_category','product category','category_name','category name'])

if not product_id_col or not category_col:
    st.error(f'Product ID or Category column not found. Columns: {list(products.columns)}')
    st.stop()

def category_name(x):
    return re.sub(r'[_\-]+', ' ', str(x).strip()).title()

def product_path(category):
    return f'ERP → Inventory → Products → {category_name(category)}'

text_cols = products.select_dtypes(include=['object','string']).columns.tolist()
if category_col not in text_cols:
    text_cols.append(category_col)
products['_search_text'] = products[text_cols].fillna('').astype(str).agg(' '.join, axis=1).str.lower()

@st.cache_resource
def build_engine(data):
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1,2), sublinear_tf=True)
    mat = vec.fit_transform(data)
    return vec, mat

vectorizer, matrix = build_engine(tuple(products['_search_text'].tolist()))

def intent(q):
    q = q.lower().strip()
    if any(x in q for x in ['payment','payments','paid','payment method','payment type','transaction','transactions']):
        return 'payment'
    if any(x in q for x in ['not delivered','undelivered','delivered','delivery','cancelled','canceled','order','orders']):
        return 'order'
    if any(x in q for x in ['customer','customers','buyer','buyers']):
        return 'customer'
    return 'product'

def payment_search(q):
    if payments.empty:
        return pd.DataFrame()
    # Generic payment query means SHOW PAYMENT RECORDS.
    if q.lower().strip() in ['payment','payments','paid','transaction','transactions']:
        return payments.head(5).copy()
    cols = payments.select_dtypes(include=['object','string']).columns.tolist()
    if not cols:
        return payments.head(5).copy()
    text = payments[cols].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
    mask = text.str.contains(re.escape(q.lower().strip()), regex=True, na=False)
    found = payments[mask]
    return (found if not found.empty else payments).head(5).copy()

def order_search(q):
    if orders.empty:
        return pd.DataFrame()
    r = orders.copy(); q = q.lower()
    delivered = find_col(r, ['order_delivered_timestamp','delivered_timestamp','delivery_date'])
    status = find_col(r, ['order_status','status'])
    if ('not delivered' in q or 'undelivered' in q) and delivered:
        r = r[r[delivered].isna()]
    elif 'delivered' in q and delivered:
        r = r[r[delivered].notna()]
    elif ('cancelled' in q or 'canceled' in q) and status:
        r = r[r[status].astype(str).str.lower().isin(['cancelled','canceled'])]
    return r.head(5)

def customer_search(q):
    if customers.empty:
        return pd.DataFrame()
    r = customers.copy()
    cols = r.select_dtypes(include=['object','string']).columns.tolist()
    if not cols:
        return r.head(5)
    text = r[cols].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
    mask = text.str.contains(re.escape(q.lower().strip()), regex=True, na=False)
    if mask.any():
        r = r[mask]
    return r.head(5)

def product_search(q):
    q = q.lower().strip()
    scores = cosine_similarity(vectorizer.transform([q]), matrix).flatten()
    r = products.copy(); r['relevance_score'] = scores
    direct = r['_search_text'].str.contains(re.escape(q), regex=True, na=False)
    r.loc[direct, 'relevance_score'] += 0.10
    r = r[r['relevance_score'] > 0].sort_values('relevance_score', ascending=False)
    return r.head(5)

def payment_output(q, r):
    lines = ['='*58, f'SEARCH QUERY: {q}', '='*58, '']
    if r.empty:
        lines.append('No payment records found.')
        return '\n'.join(lines)
    pid = find_col(r, ['payment_id','paymentid','payment id','id'])
    oid = find_col(r, ['order_id','orderid','order id'])
    ptype = find_col(r, ['payment_type','payment type','payment_method','method'])
    value = find_col(r, ['payment_value','payment value','amount','value'])
    for i, (_, row) in enumerate(r.iterrows(), 1):
        lines += [f'Result {i}', '-'*50, '']
        if pid: lines.append(f'Payment ID      : {row[pid]}')
        if oid: lines.append(f'Order ID        : {row[oid]}')
        if ptype: lines.append(f'Payment Type    : {row[ptype]}')
        if value: lines.append(f'Payment Value   : {row[value]}')
        lines += ['Exact ERP Path  : ERP → Sales → Payments', '']
    return '\n'.join(lines)

def product_output(q, r):
    lines = ['='*58, f'SEARCH QUERY: {q}', '='*58, '']
    if r.empty:
        lines.append('No relevant product results found.')
        return '\n'.join(lines)
    for i, (_, row) in enumerate(r.iterrows(), 1):
        lines += [f'Result {i}', '-'*50, '',
                  f'Product ID      : {row[product_id_col]}',
                  f'Category        : {row[category_col]}',
                  f'Relevance Score : {min(float(row["relevance_score"]),1.0):.2f}',
                  f'Exact ERP Path  : {product_path(row[category_col])}', '']
    return '\n'.join(lines)

def generic_output(q, r, kind):
    lines = ['='*58, f'SEARCH QUERY: {q}', '='*58, '']
    if r.empty:
        lines.append('No relevant results found.')
        return '\n'.join(lines)
    path = 'ERP → Sales → Orders' if kind == 'order' else 'ERP → Customers'
    for i, (_, row) in enumerate(r.iterrows(), 1):
        lines += [f'Result {i}', '-'*50, '']
        for c in r.columns:
            lines.append(f'{str(c):<18}: {row[c]}')
        lines += [f'Exact ERP Path  : {path}', '']
    return '\n'.join(lines)

query = st.text_input('💬 Enter your search query:', placeholder='watch / payment / delivered orders / customer')

if st.button('🔍 Search', use_container_width=True):
    if not query.strip():
        st.warning('Please enter a search query.')
    else:
        kind = intent(query)
        if kind == 'payment':
            result = payment_search(query)
            output = payment_output(query, result)
        elif kind == 'order':
            result = order_search(query)
            output = generic_output(query, result, 'order')
        elif kind == 'customer':
            result = customer_search(query)
            output = generic_output(query, result, 'customer')
        else:
            result = product_search(query)
            output = product_output(query, result)
        st.subheader('🔎 Search Results')
        st.code(output, language='text')
        st.success(f'Detected Intent: {kind} | Results: {len(result)}')
