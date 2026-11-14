"""Project-wide constants: paths, column roles, random seed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "fraud_oracle.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
METRICS_DIR = ROOT / "reports" / "metrics"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42
TARGET = "FraudFound_P"
TEST_SIZE = 0.20
N_SPLITS = 5

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]

# Columns that are never used as model inputs, with the reason.
DROPPED_COLUMNS = {
    "PolicyNumber": "row identifier (1..n), carries no information about the claim",
    "Year": "calendar year cannot generalise to future claims; kept only for out-of-time validation",
    "PolicyType": "fully determined by VehicleCategory x BasePolicy",
    "Age": "0 is a hidden missing code (320 rows) and the binned AgeOfPolicyHolder is complete",
}

# Binary columns and the level that is encoded as 1.
BINARY_FEATURES = {
    "AccidentArea": "Rural",
    "Sex": "Female",
    "Fault": "Third Party",
    "PoliceReportFiled": "Yes",
    "WitnessPresent": "Yes",
    "AgentType": "Internal",
}

# Ordered categorical columns with their natural order (low -> high).
ORDINAL_FEATURES = {
    "VehiclePrice": ["less than 20000", "20000 to 29000", "30000 to 39000",
                     "40000 to 59000", "60000 to 69000", "more than 69000"],
    "Days_Policy_Accident": ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"],
    "Days_Policy_Claim": ["8 to 15", "15 to 30", "more than 30"],
    "PastNumberOfClaims": ["none", "1", "2 to 4", "more than 4"],
    "AgeOfVehicle": ["new", "2 years", "3 years", "4 years", "5 years",
                     "6 years", "7 years", "more than 7"],
    "AgeOfPolicyHolder": ["16 to 17", "18 to 20", "21 to 25", "26 to 30", "31 to 35",
                          "36 to 40", "41 to 50", "51 to 65", "over 65"],
    "NumberOfSuppliments": ["none", "1 to 2", "3 to 5", "more than 5"],
    "AddressChange_Claim": ["no change", "under 6 months", "1 year",
                            "2 to 3 years", "4 to 8 years"],
    "NumberOfCars": ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"],
    "Deductible": [300, 400, 500, 700],
    "DriverRating": [1, 2, 3, 4],
    "WeekOfMonth": [1, 2, 3, 4, 5],
    "WeekOfMonthClaimed": [1, 2, 3, 4, 5],
    "ClaimLagMonths": [0, 1, 2, 3],
}

# Unordered categorical columns.
NOMINAL_FEATURES = [
    "Make", "MaritalStatus", "VehicleCategory", "BasePolicy", "RepNumber",
    "Month", "DayOfWeek", "MonthClaimed", "DayOfWeekClaimed",
]

FEATURES = list(BINARY_FEATURES) + list(ORDINAL_FEATURES) + NOMINAL_FEATURES

# Attributes that are legally or ethically sensitive in insurance decisions.
SENSITIVE_FEATURES = ["Sex", "MaritalStatus", "AgeOfPolicyHolder"]
