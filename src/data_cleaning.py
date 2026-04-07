def clean_data(df):
    
    df = df.drop_duplicates()

    df['Attrition'] = df['Attrition'].map({
        "Yes": 1,
        "No": 0
    })

    df['OverTime'] = df['OverTime'].map({
        "Yes": 1,
        "No": 0
    })

    return df