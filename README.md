# AI ERP Smart Search - Streamlit App

## Folder structure

Keep the project like this:

AI_ERP_Search/
├── app.py
├── requirements.txt
├── dataset/
│   ├── your ERP CSV files
│   └── ...
└── notebook/
    └── your Jupyter notebook

## Run

Open Command Prompt in the AI_ERP_Search folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app automatically looks for CSV files inside `dataset/`.

It supports common Olist/ERP file names such as:
- olist_customers_dataset.csv
- olist_orders_dataset.csv
- olist_order_items_dataset.csv
- olist_products_dataset.csv
- olist_payments_dataset.csv

It also searches by keywords, so names such as customers.csv, orders.csv, products.csv, payments.csv and order_items.csv work too.

## AI model

Semantic fallback uses the pretrained Sentence Transformer:

all-MiniLM-L6-v2

The model is not trained from scratch in the app. It converts text into embeddings and uses cosine similarity for semantic search.

## Product images

If your product table contains an `image_path`, `product_image`, `image`, or `img_path` column and the paths point to existing local files, the app will show product previews automatically.
