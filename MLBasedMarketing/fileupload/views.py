from django.shortcuts import render
from .forms import UploadFileForm
from .ml_model import load_model
import csv
from django.http import HttpResponse
from django.http import HttpResponseRedirect

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer
from sklearn.decomposition import PCA
from collections import defaultdict
import numpy as np
import itertools
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go

from io import BytesIO
import base64

# Create your views here.

def prepare_dataset(df):
    df.rename(columns={"Торговый день": "trading_day", "ID Клиента": "client_id",
                       "Регион проживания": "region", "МАГАЗИН": "shop",
                       "Тип карточки (B-VISA, S-MASTERCARD или МИР)": "card_type", "Сумма расходов": "spendings",
                       "Предоставленная скидка": "sale", "Год выдачи паспорта": "passport", "Возраст клиента": "age"},
              inplace=True)

    df.loc[df["sale"] == "скидки не было", "sale"] = 0
    df.loc[df["sale"] == "01", "sale"] = 1
    df.loc[df["sale"] == "02", "sale"] = 2
    df.loc[df["sale"] == "03", "sale"] = 3
    df.loc[df["sale"] == "04", "sale"] = 4
    df.loc[df["sale"] == "05", "sale"] = 5
    df.loc[df["sale"] == "06", "sale"] = 6
    df.loc[df["sale"] == "07", "sale"] = 7
    df.loc[df["sale"] == "08", "sale"] = 8
    df.loc[df["sale"] == "09", "sale"] = 9

    df['sale'] = df['sale'].astype(int)

    def aggregate_data(df):
        # Define the aggregation dictionary
        agg_dict = {
            'trading_day': 'first',
            'region': 'first',
            'shop': 'first',
            'card_type': 'first',
            'passport': 'first',
            'age': 'first',
            'sale': 'first',
            'spendings': 'sum'
        }

        # Group by client_id, sort within groups by spendings, and aggregate
        grouped = df.sort_values('spendings', ascending=False).groupby('client_id').agg(agg_dict)

        return grouped

    aggregated = aggregate_data(df)

    df = df.sort_values('trading_day')

    # Frequency of visits
    frequency_of_visits = df.groupby('client_id')['trading_day'].nunique()

    # Unique shops
    unique_shops = df.groupby('client_id')['shop'].nunique()

    # Most visited shop
    most_visited_shop = df.groupby('client_id')['shop'].agg(lambda x: x.value_counts().index[0])

    # Most used card type
    most_used_card_type = df.groupby('client_id')['card_type'].agg(lambda x: x.value_counts().index[0])

    # Age group
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    aggregated['age_group'] = discretizer.fit_transform(aggregated[['age']])

    # Recency, Frequency, Monetary features
    df['trading_day'] = pd.to_datetime(df['trading_day'])
    snapshot_date = df['trading_day'].max() + pd.Timedelta(days=1)  # The next day after latest transaction
    rfm = df.groupby('client_id').agg({
        'trading_day': lambda x: (snapshot_date - x.max()).days,
        'client_id': 'count',
        'spendings': 'sum'
    }).rename(columns={
        'trading_day': 'recency',
        'client_id': 'total_number_of_purchases',
        'spendings': 'monetary'
    })

    def calculate_clv(df):
        # Total revenue from the customer
        total_revenue = df.groupby('client_id')['spendings'].sum()

        # Frequency of visits
        frequency_of_visits = df.groupby('client_id')['trading_day'].nunique()

        # CLV is total revenue times frequency of visits
        clv = total_revenue * frequency_of_visits

        return clv

    clv = calculate_clv(df)
    clv = pd.DataFrame({'clv': clv})

    # Spending volatility
    spending_volatility = df.groupby('client_id')['spendings'].std()
    spending_volatility

    # Spending trend
    df['spending_trend'] = df.groupby('client_id')['spendings'].pct_change()
    df = df.fillna(0)
    df

    aggregated_df = df.groupby('client_id')['spending_trend'].mean().reset_index()
    aggregated_df = aggregated_df.fillna(0)

    res = pd.merge(aggregated, frequency_of_visits, on="client_id")
    res = pd.merge(res, unique_shops, on="client_id")
    res = pd.merge(res, most_visited_shop, on="client_id")
    res = pd.merge(res, most_used_card_type, on="client_id")

    res.drop('trading_day_x', axis=1, inplace=True)
    res.drop('shop_x', axis=1, inplace=True)
    res.drop('card_type_x', axis=1, inplace=True)
    res.drop('age', axis=1, inplace=True)

    res.rename(columns={"spendings": "total_spendings", "trading_day_y": "frequency_of_purchases",
                        "shop_y": "unique_shops", "shop": "most_visited_shop",
                        "card_type_y": "most_used_card"},
               inplace=True)

    res = pd.merge(res, rfm, on="client_id")
    res.drop('monetary', axis=1, inplace=True)

    res = pd.merge(res, clv, on="client_id")

    res = pd.merge(res, spending_volatility, on="client_id")
    res.rename(columns={"spendings": "spending_volatility"},
               inplace=True)
    res = pd.merge(res, aggregated_df, on="client_id")
    res.drop('client_id', axis=1, inplace=True)

    features = ['region', 'most_visited_shop']

    res[features] = res[features].astype(str)

    for feat in features:
        lbe = LabelEncoder()
        lbe.fit(res[feat].values)
        diz_map_train = dict(zip(lbe.classes_, lbe.transform(lbe.classes_) + 1))
        res[feat] = [diz_map_train[i] for i in res[feat].values]

    ### CREATE ENCODING FOR EACH CATEGORICAL FEATURES ###

    diz_enc = {}

    for g_f in features:

        passive_feat = features[:]
        passive_feat.remove(g_f)

        diz_group = defaultdict(list)
        for f in passive_feat:

            #         print('--- groupby:', g_f, '###', f, '---')
            data = res.copy()
            data.reset_index(inplace=True, level=None)

            group = data.groupby([g_f, f])['index'].count()
            group = group.unstack().fillna(0)

            ent = group.apply(entropy, axis=1)
            for i, e in ent.to_dict().items():
                diz_group[i].extend([e])

            diz_group[0] = [.0] * len(passive_feat)

        diz_enc[g_f] = diz_group

    ### MAP CLASSES TO VECTORS ###

    train_feat = res.copy()

    for f in features:
        train_feat[f] = train_feat[f].map(diz_enc[f])

    train_feat = [list(itertools.chain(*raw[1:])) for raw in train_feat[features].itertuples()]

    arr = np.array(train_feat)

    res['cat_feat_1'] = arr[:, 0]
    res['cat_feat_2'] = arr[:, 1]

    res.drop('most_visited_shop', axis=1, inplace=True)
    res.drop('region', axis=1, inplace=True)

    one_hot_train = pd.get_dummies(res['most_used_card'], drop_first=True)
    res = pd.concat((res.drop('most_used_card', axis=1), one_hot_train), axis=1)

    res.drop_duplicates(inplace=True)

    res = res.fillna(0)

    res['clv'] = np.log(res['clv'])
    res['total_spendings'] = np.log(res['total_spendings'])
    res = res.fillna(0)

    res_csv = res.to_csv('res_csv.csv', index_label=False)

    scaler = StandardScaler()
    scaler.fit(res)
    res_pca = scaler.transform(res)

    def determine_num_components(res_pca):
        pca = PCA(n_components=None)
        pca.fit(res_pca)

        exp_var = pca.explained_variance_ratio_ * 100
        cum_exp_var = np.cumsum(exp_var)

        num_components = np.argmax(cum_exp_var >= 80) + 1

        return num_components

    num_components = determine_num_components(res_pca)
    pca = PCA(n_components=num_components)

    pca_dataset = pca.fit_transform(res_pca)

    column_names = [f'PCA_{i + 1}' for i in range(num_components)]
    pca_dataset = pd.DataFrame(pca_dataset, columns=column_names)

    X = pca_dataset

    return X


def final_data(df, X):
    df.rename(columns={"Торговый день": "trading_day", "ID Клиента": "client_id",
                       "Регион проживания": "region", "МАГАЗИН": "shop",
                       "Тип карточки (B-VISA, S-MASTERCARD или МИР)": "card_type", "Сумма расходов": "spendings",
                       "Предоставленная скидка": "sale", "Год выдачи паспорта": "passport", "Возраст клиента": "age"},
              inplace=True)

    agg_dict = {
        'trading_day': 'first',
        'region': 'first',
        'shop': 'first',
        'card_type': 'first',
        'passport': 'first',
        'age': 'first',
        'sale': 'first',
        'spendings': 'sum'
    }

    # Group by client_id, sort within groups by spendings, and aggregate
    grouped = df.sort_values('spendings', ascending=False).groupby('client_id').agg(agg_dict)
    grouped.reset_index(inplace=True)

    grouped = grouped[['client_id']]  # keep only client_id column in df1
    X = X[['cluster']]  # keep only cluster column in df2

    # ensure both dataframes have the same number of rows
    assert len(grouped) == len(
        X), "DataFrames have different lengths. They must be the same length to concatenate column-wise."

    grouped.reset_index(drop=True, inplace=True)  # reset index in place
    X.reset_index(drop=True, inplace=True)  # reset index in place

    res = pd.concat([grouped, X], axis=1)  # concatenate dataframes column-wise

    return res


def handle_uploaded_file(f):
    with open('some_file.csv', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return 'some_file.csv'


def upload(request):
    file_uploaded = False
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            filepath = handle_uploaded_file(request.FILES['file'])

            file_uploaded = True

            # Load the trained model
            model = load_model()

            # Load the uploaded file into a DataFrame
            df = pd.read_csv(filepath)

            # Prepare the dataset

            df_prepared = prepare_dataset(df)

            # Use the model to make predictions
            predictions = model.predict(df_prepared)

            # Convert to list
            predictions = predictions.tolist()

            # Add the predictions to the DataFrame
            df_prepared['cluster'] = predictions

            df_final = final_data(df, df_prepared)

            # Select only the first 50 rows for table
            df_head = df_final.head(50)

            # Convert the DataFrame to HTML
            df_html = df_head.to_html(index=True)

            # Save the DataFrame as a CSV file
            df_final.to_csv('updated_file.csv', index=False)
            df_prepared.to_csv('prepared_file.csv', index=False)

            # Generate the graphs
            pie = piechart(df_prepared)
            bar = barchart(df_prepared)

            # Here you can process the predictions and pass them to your template
            return render(request, 'upload.html', {'form': form, 'predictions': predictions, 'df_html': df_html, 'bar': bar, 'pie': pie, 'file_uploaded': file_uploaded})
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})

def piechart(df):
    pie = df.groupby('cluster').size().reset_index()
    pie.columns = ['cluster', 'value']

    fig = go.Figure(data=[go.Pie(labels=pie['cluster'], values=pie['value'], marker_colors=['blue', 'red', 'green'])])

    fig.update_layout(title='Cluster Distribution PieChart')
    fig.update_layout(autosize=True, width=500, height=500)
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)

    fig_html = fig.to_html(fig, full_html=False)

    return fig_html


def barchart(df):
    bar = df.groupby('cluster').size().reset_index()
    bar.columns = ['cluster', 'value']

    fig = go.Figure(data=[go.Bar(x=bar['cluster'], y=bar['value'], marker_color=['blue', 'red', 'green'])])

    fig.update_layout(title='Cluster Distribution BarChart',
                      xaxis_title='Cluster',
                      yaxis_title='Count')

    fig.update_layout(autosize=True, width=500, height=500)

    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)

    fig_html = fig.to_html(fig, full_html=False)

    return fig_html

def download_file(request):
    # Open the Updated CSV file
    with open('updated_file.csv', 'r') as file:
        response = HttpResponse(file.read(), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="updated_file.csv"'
    return response


def calculate_cluster_means(df_final, res_csv):
    # Group the dataframe by 'cluster' column and calculate the mean for each feature
    df = pd.concat([df_final, res_csv], axis=1)
    cluster_means = df.groupby('cluster').mean()
    cluster_means.drop(['client_id'], axis=1, inplace=True)

    return cluster_means


def statistics(request):
    df_final = pd.read_csv('updated_file.csv')
    df_prepared = pd.read_csv('prepared_file.csv')
    df_res = pd.read_csv('res_csv.csv')

    # Merge the datasets based on common columns
    merged_df = pd.merge(df_res, df_prepared, left_index=True, right_index=True)
    # Get the columns for merged df, which will be used to choose the columns to draw the scatter plot by
    df_columns = merged_df.columns

    if 'plot' not in request.session:
        request.session['plot'] = ''
    plot = request.session['plot']

    if 'df_html' not in request.session:
        request.session['df_html'] = ''
    df_html = request.session['df_html']

    if 'x_column' in request.session:
        x_column = request.session['x_column']
    else:
        x_column = 'passport'
    if 'y_column' in request.session:
        y_column = request.session['y_column']
    else:
        y_column = 'passport'
    if 'selected_df' in request.session:
        selected_df = request.session['selected_df']
    else:
        selected_df = 'Mean'

    if request.method == 'POST':
        lst = merge_and_plot_scatter(df_res, df_prepared, request)
        plot = lst[2]
        x_column = lst[0]
        y_column = lst[1]
        request.session['plot'] = plot
        lst2 = show_desired_table(df_final, df_res, request)
        try:
            selected_df = lst2[0]
        except KeyError:
            selected_df = lst[-1]
        df_html = lst2[-1]
        request.session['df_html'] = df_html
        request.session['selected_df'] = selected_df

    context = {
        'df_columns': df_columns,
        'plot': plot,
        'df_html': df_html,
        'selected_df': selected_df,
        'x_column': x_column,
        'y_column': y_column
    }

    return render(request, 'statistics.html', context)

def merge_and_plot_scatter(res, df_prepared, request):
    # Merge the datasets based on common columns
    if request.method == 'POST':

        if 'scatter' in request.POST:

            x_column = request.POST.get('x_column')
            y_column = request.POST.get('y_column')

            lst = []
            merged_df = pd.merge(res, df_prepared, left_index=True, right_index=True)
            # Set the color palette for the clusters
            cluster_colors = {0: 'blue', 1: 'red', 2: 'green'}

            # Plot the scatterplot
            plt.figure(figsize=(10, 8))
            sns.scatterplot(data=merged_df, x=x_column, y=y_column, hue='cluster', palette=cluster_colors)

            # Add axis labels and a title
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title('Scatterplot with Cluster Colors')

            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            image_png = buf.getvalue()
            buf.close()
            plot = base64.b64encode(image_png).decode('utf-8')

            request.session['plot'] = plot
            request.session['x_column'] = x_column
            request.session['y_column'] = y_column

            lst.append(x_column)
            lst.append(y_column)
            lst.append(plot)
            return lst

        else:
            if 'plot' in request.session:
                lst = []
                plot = request.session['plot']
                try:
                    y_column = request.session['y_column']
                except KeyError:
                    y_column = 'passport'
                try:
                    x_column = request.session['x_column']
                except KeyError:
                    x_column = 'passport'

                lst.append(x_column)
                lst.append(y_column)
                lst.append(plot)
                return lst

def show_desired_table(df_final, df_res, request):

    if request.method == 'POST':

        lst = []

        if 'load_df' in request.POST:

            selected_df = request.POST.get('selected_df')

            if selected_df == 'merged':
                # Merged final and res dataframes
                merged_df2 = pd.concat([df_final, df_res], axis=1)
                df_html = merged_df2.to_html(index=True)
                request.session['selected_df'] = selected_df
            elif selected_df == 'describe':
                df_html = df_res.describe().to_html(index=True)
                request.session['selected_df'] = selected_df
            elif selected_df == 'mean':
                mean = calculate_cluster_means(df_final, df_res)
                df_html = mean.to_html(index=True)
                request.session['selected_df'] = selected_df
        else:
            if 'df_html' in request.session:
                try:
                    selected_df = request.session['selected_df']
                except KeyError:
                    selected_df = 'mean'
                df_html = request.session['df_html']

        lst.append(selected_df)
        lst.append(df_html)

        return lst

# def del_sessions(request):
#     if 'df_html' in request.session:
#         del request.session['df_html']
#     if 'plot' in request.session:
#         del request.session['plot']
#     if 'x_column' in request.session:
#         del request.session['x_column']
#     if 'y_column' in request.session:
#         del request.session['y_column']
#     if 'selected_df' in request.session:
#         del request.session['selected_df']
#
#     return HttpResponseRedirect('/statistics')