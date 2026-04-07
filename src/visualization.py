import plotly.express as px


def plot_department_attrition(df):

    fig = px.bar(
        df,
        x="Department",
        y="Attrition",
        title="Attrition Rate by Department"
    )

    return fig


def plot_jobrole_attrition(df):

    fig = px.bar(
        df,
        x="JobRole",
        y="Attrition",
        title="Attrition by Job Role"
    )

    return fig