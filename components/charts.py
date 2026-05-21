import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

COLOR_MAP = {
    'Yes': '#f97316',
    'No': '#22c55e',
    'Month-to-month': '#38bdf8',
    'One year': '#6366f1',
    'Two year': '#8b5cf6',
    'DSL': '#a855f7',
    'Fiber optic': '#ec4899',
    'None': '#64748b',
}


def churn_donut(df: pd.DataFrame):
    fig = px.pie(
        df,
        names='churn',
        hole=0.45,
        title='Retention vs Churn',
        color='churn',
        color_discrete_map={'Yes': COLOR_MAP['Yes'], 'No': COLOR_MAP['No']}
    )
    fig.update_traces(textinfo='percent+label', pull=[0.05, 0], marker=dict(line=dict(color='white', width=2)))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


def contract_bar(df: pd.DataFrame):
    fig = px.bar(
        df,
        x='contract',
        color='churn',
        title='Contract Mix by Customer Status',
        barmode='group',
        color_discrete_map={'Yes': COLOR_MAP['Yes'], 'No': COLOR_MAP['No']}
    )
    fig.update_layout(
        xaxis_title='Contract Type',
        yaxis_title='Customer Count',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(title='Churn', orientation='h', yanchor='bottom', y=1.02, x=0.5, xanchor='center')
    )
    return fig


def tenure_scatter(df: pd.DataFrame):
    fig = px.scatter(
        df,
        x='tenure',
        y='monthly_charges',
        color='churn',
        size='support_calls' if 'support_calls' in df.columns else None,
        title='Tenure vs Monthly Charge',
        hover_data=['contract', 'internet_service', 'payment_method'],
        color_discrete_map={'Yes': COLOR_MAP['Yes'], 'No': COLOR_MAP['No']},
        trendline='ols',
        template='plotly_white'
    )
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color='white')))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def internet_service_count(df: pd.DataFrame):
    if 'internet_service' not in df.columns:
        return go.Figure()
    fig = px.histogram(
        df,
        x='internet_service',
        color='churn',
        title='Internet Service Adoption by Churn Status',
        barmode='group',
        color_discrete_map={'Yes': COLOR_MAP['Yes'], 'No': COLOR_MAP['No']}
    )
    fig.update_layout(
        xaxis_title='Internet Service',
        yaxis_title='Count',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(title='Churn', orientation='h', yanchor='bottom', y=1.02, x=0.5, xanchor='center')
    )
    return fig


def monthly_charges_histogram(df: pd.DataFrame):
    fig = px.histogram(
        df,
        x='monthly_charges',
        nbins=30,
        color='churn',
        opacity=0.8,
        title='Monthly Charges Distribution',
        color_discrete_map={'Yes': COLOR_MAP['Yes'], 'No': COLOR_MAP['No']}
    )
    fig.update_layout(
        xaxis_title='Monthly Charge ($)',
        yaxis_title='Customer Count',
        bargap=0.15,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


def tenure_box_plot(df: pd.DataFrame):
    fig = px.box(
        df,
        x='churn',
        y='tenure',
        color='churn',
        title='Tenure Distribution by Churn',
        color_discrete_map={'Yes': COLOR_MAP['Yes'], 'No': COLOR_MAP['No']}
    )
    fig.update_layout(
        xaxis_title='Churn',
        yaxis_title='Tenure (months)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig


def risk_gauge(probability: float):
    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=probability * 100,
        number={'suffix': '%'},
        delta={'reference': 50, 'increasing': {'color': '#fb7185'}, 'decreasing': {'color': '#22c55e'}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#38bdf8'},
            'bgcolor': '#e2e8f0',
            'steps': [
                {'range': [0, 33], 'color': '#22c55e'},
                {'range': [33, 66], 'color': '#facc15'},
                {'range': [66, 100], 'color': '#fb7185'}
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        },
        title={'text': 'Churn Risk Score', 'font': {'size': 18}}
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def correlation_heatmap(df: pd.DataFrame):
    numeric = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='aggrnyl',
        aspect='auto',
        title='Numeric Feature Correlation'
    )
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig
