"""Official Feature Dictionary and Metadata for CDC Diabetes Health Indicators.

Source: Behavioral Risk Factor Surveillance System (BRFSS) 2015 / UCI Dataset #891.
Documentation: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
Author: Abhiram (AI/ML Engineer)
"""

from typing import Dict, Any

CDC_FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    "Diabetes_binary": {
        "role": "Target",
        "type": "Binary Categorical",
        "values": {0: "No diabetes (includes prediabetes in binary split)", 1: "Diabetes"},
        "description": "Whether the respondent has been diagnosed with diabetes.",
    },
    "Diabetes_012": {
        "role": "Target (Multiclass)",
        "type": "Multiclass Categorical",
        "values": {0: "No diabetes", 1: "Prediabetes", 2: "Diabetes"},
        "description": "3-class classification of diabetes status.",
    },
    "HighBP": {
        "role": "Feature",
        "type": "Binary Categorical",
        "values": {0: "No high BP", 1: "High BP diagnosed"},
        "description": "Ever told by a doctor, nurse, or other health professional that blood pressure is high.",
    },
    "HighChol": {
        "role": "Feature",
        "type": "Binary Categorical",
        "values": {0: "No high cholesterol", 1: "High cholesterol diagnosed"},
        "description": "Ever told by a doctor or health professional that blood cholesterol is high.",
    },
    "CholCheck": {
        "role": "Feature",
        "type": "Binary Categorical",
        "values": {0: "No cholesterol check in past 5 years", 1: "Cholesterol checked in past 5 years"},
        "description": "Cholesterol check performed within the past 5 years.",
    },
    "BMI": {
        "role": "Feature",
        "type": "Continuous / Integer",
        "values": "Range: 12 to 98 kg/m^2",
        "description": "Body Mass Index calculated as weight (kg) / [height (m)]^2.",
    },
    "Smoker": {
        "role": "Feature",
        "type": "Binary Categorical",
        "values": {0: "No", 1: "Yes"},
        "description": "Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes].",
    },
    "Stroke": {
        "role": "Feature",
        "type": "Binary Categorical",
        "values": {0: "No", 1: "Yes"},
        "description": "Ever told by a doctor that you had a stroke.",
    },
    "HeartDiseaseorAttack": {
        "role": "Feature",
        "type": "Binary Categorical",
        "values": {0: "No", 1: "Yes"},
        "description": "Reported coronary heart disease (CHD) or myocardial infarction (MI).",
    },
    "PhysActivity": {
        "role": "Feature",
        "type": "Binary Categorical",
        "values": {0: "No", 1: "Yes"},
        "description": "Physical activity or exercise during the past 30 days other than regular job.",
    },
    "Fruits": {
        "role": "Feature",
        "type": "Binary Categorical",
        "values": {0: "No", 1: "Yes"},
        "description": "Consume fruit 1 or more times per day.",
    },
    "Veggies": {
        "role": "Feature",
        "type": "Binary Categorical",
        "values": {0: "No", 1: "Yes"},
        "description": "Consume beans or dark green vegetables 1 or more times per day.",
    },
    "HvyAlcoholConsump": {
        "role": "Feature",
        "type": "Binary Categorical",
        "values": {0: "No", 1: "Yes"},
        "description": "Heavy drinkers: adult men >= 14 drinks/week; adult women >= 7 drinks/week.",
    },
    "AnyHealthcare": {
        "role": "Feature",
        "type": "Binary Categorical",
        "values": {0: "No", 1: "Yes"},
        "description": "Have any kind of health care coverage (insurance, HMOs, Medicare, etc.).",
    },
    "NoDocbcCost": {
        "role": "Feature",
        "type": "Binary Categorical",
        "values": {0: "No", 1: "Yes"},
        "description": "Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?",
    },
    "GenHlth": {
        "role": "Feature",
        "type": "Ordinal Categorical (1-5)",
        "values": {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"},
        "description": "Self-reported general health evaluation on a 5-point Likert scale.",
    },
    "MentHlth": {
        "role": "Feature",
        "type": "Discrete Integer (0-30)",
        "values": "Range: 0 to 30 days",
        "description": "Number of days during past 30 days where mental health (stress, depression, problems with emotions) was not good.",
    },
    "PhysHlth": {
        "role": "Feature",
        "type": "Discrete Integer (0-30)",
        "values": "Range: 0 to 30 days",
        "description": "Number of days during past 30 days where physical health (illness, injury) was not good.",
    },
    "DiffWalk": {
        "role": "Feature",
        "type": "Binary Categorical",
        "values": {0: "No", 1: "Yes"},
        "description": "Serious difficulty walking or climbing stairs.",
    },
    "Sex": {
        "role": "Feature",
        "type": "Binary Categorical",
        "values": {0: "Female", 1: "Male"},
        "description": "Reported biological sex.",
    },
    "Age": {
        "role": "Feature",
        "type": "Ordinal Categorical (1-13)",
        "values": {
            1: "Age 18 to 24",
            2: "Age 25 to 29",
            3: "Age 30 to 34",
            4: "Age 35 to 39",
            5: "Age 40 to 44",
            6: "Age 45 to 49",
            7: "Age 50 to 54",
            8: "Age 55 to 59",
            9: "Age 60 to 64",
            10: "Age 65 to 69",
            11: "Age 70 to 74",
            12: "Age 75 to 79",
            13: "Age 80 or older",
        },
        "description": "Thirteen-level age category from young adult (18-24) to elderly (80+).",
    },
    "Education": {
        "role": "Feature",
        "type": "Ordinal Categorical (1-6)",
        "values": {
            1: "Never attended school or only kindergarten",
            2: "Grades 1 through 8 (Elementary)",
            3: "Grades 9 through 11 (Some high school)",
            4: "Grade 12 or GED (High school graduate)",
            5: "College 1 to 3 years (Some college or technical school)",
            6: "College 4 years or more (College graduate)",
        },
        "description": "Highest level of school completed.",
    },
    "Income": {
        "role": "Feature",
        "type": "Ordinal Categorical (1-8)",
        "values": {
            1: "Less than $10,000",
            2: "$10,000 to <$15,000",
            3: "$15,000 to <$20,000",
            4: "$20,000 to <$25,000",
            5: "$25,000 to <$35,000",
            6: "$35,000 to <$50,000",
            7: "$50,000 to <$75,000",
            8: "$75,000 or more",
        },
        "description": "Annual household income bracket from all sources.",
    },
}

BINARY_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"
]

ORDINAL_FEATURES = ["GenHlth", "Age", "Education", "Income"]

CONTINUOUS_NUMERICAL_FEATURES = ["BMI", "MentHlth", "PhysHlth"]
