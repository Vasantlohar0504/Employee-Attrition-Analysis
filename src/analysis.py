def attrition_by_department(df):
    
    return df.groupby("Department")["Attrition"].mean().reset_index()


def attrition_by_jobrole(df):

    return df.groupby("JobRole")["Attrition"].mean().reset_index()


def salary_attrition(df):

    return df.groupby("Attrition")["MonthlyIncome"].mean()