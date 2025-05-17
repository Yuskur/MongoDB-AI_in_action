import pandas as pd

def clean_data():
    df = pd.read_csv("student-mental-health.csv")

    # ======================================================== DATA CLEANING =======================================================
    #Set gender columns to lowercase (I found some male and Male)
    df["Choose your gender"] = df["Choose your gender"].str.lower()
    df["Choose your gender"] = df["Choose your gender"].replace({"male": 1, "female": 0})

    #Change the name of Choose your gender column to gender for simplicity
    df.rename(columns={"Choose your gender": "Gender"}, inplace=True)

    df["Your current year of Study"] = df["Your current year of Study"].str.lower()
    df["Your current year of Study"] = df["Your current year of Study"].replace({
        "year 1": 1,
        "year 2": 2,
        "year 3": 3,
        "year 4": 4,
    })

    # get the avg of the student gpa
    def gpa_average(gpa_range):
        gpa_list = gpa_range.split(" - ")
        gpa_avg = (float(gpa_list[0]) + float(gpa_list[1])) / 2
        return gpa_avg

    df["What is your CGPA?"] = df["What is your CGPA?"].map(gpa_average)

    df["Marital status"] = df["Marital status"].str.lower()
    df["Marital status"] = df["Marital status"].replace({"yes": 1, "no": 0})

    # change mental health stats to 1 and 0
    df[[
        "Do you have Depression?", 
        "Do you have Anxiety?", 
        "Do you have Panic attack?", 
        "Did you seek any specialist for a treatment?"
    ]] = df[[
        "Do you have Depression?",
        "Do you have Anxiety?", 
        "Do you have Panic attack?", 
        "Did you seek any specialist for a treatment?"]].replace({"Yes": 1, "No": 0})


    # We don't really need time stamps for the model so we will remove it
    df.drop(columns=["Timestamp"], inplace=True)

    # Add a new column called "Mental health risk" that is 1 if any mh columns are 1 and 0 if not
    df["Mental health risk"] = df[[
        "Do you have Depression?", 
        "Do you have Anxiety?", 
        "Do you have Panic attack?", 
        "Did you seek any specialist for a treatment?"
    ]].any(axis=1).replace({True: 1, False: 0})

    return df