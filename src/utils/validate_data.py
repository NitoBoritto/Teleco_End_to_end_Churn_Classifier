import great_expectations as ge
from typing import Tuple, List

def validate_telco_data(df) -> Tuple[bool, list[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    
    """
    print("🔎 Starting data validation check with Great Expections...")
    
    # Convert pandas df to Great Expections dataset
    ge_df = ge.dataset.PandasDataset(df)
    
    # 1️⃣ Schema validation
    print("💬 Validating schema and required columns...")
    
    # Customer identifier must exist
    ge_df.expect_column_to_exist("customerID")
    ge_df.expect_column_values_to_not_be_null("customerID")
    
    # Core demographic features
    ge_df.expect_column_to_exist("gender")
    ge_df.expect_column_to_exist("Partner")
    ge_df.expect_column_to_exist("Dependents")
    
    # Service features
    ge_df.expect_column_to_exist("PhoneService")
    ge_df.expect_column_to_exist("InternetService")
    ge_df.expect_column_to_exist("Contract")
    
    # Financial features
    ge_df.expect_column_to_exist("tenure")
    ge_df.expect_column_to_exist("MonthlyCharges")
    ge_df.expect_column_to_exist("TotalCharges")
    
    # 2️⃣ Business logic validation
    print("💬 Validation business logic constraints")
    
    # Gender data integrity
    ge_df.expect_column_values_to_be_in_set('gender', ['Male', 'Female'])
    
    # Yes/No fields data integreity
    ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])
    
    # Contract types must be valid
    ge_df.expect_column_values_to_be_in_set(
        'Contract',
        ['Month-to_Month', 'One year', 'Two year']
    )
    # Internet service types
    ge_df.expect_column_values_to_be_in_set(
        'InternetService',
        ['DSL', 'Fiber optic', 'No']
    )
    
    # 3️⃣ Numeric Validations
    print('💬 Validating numeric ranges and business constraints...')
    
    # Tenure must be non-negative
    ge_df.expect_column_values_to_be_between('tenure', min_value = 0)
    
    # Monthly charges must be positive
    ge_df.expect_column_values_to_be_between('MonthlyCharges', min_value = 0)
    
    # 4️⃣ Statistical Validation
    print('💬 Validating statistical properties...')
    
    # Tenure should be reasonable (max ~10 years = 120 months for telecom)
    ge_df.expect_column_values_to_be_between('tenure', min_value = 0, max_value = 120)
    
    # Monthly charges should be within reasonable business range
    ge_df.expect_column_values_to_be_between('MonthlyCharges', min_value = 0, max_value = 200)
    
    # No missing values in critical numeric featuers
    ge_df.expect_column_values_to_not_be_null('tenure')
    ge_df.expect_column_values_to_not_be_null('MonthlyCharges')
    
    # 5️⃣ Data Consistency
    print('💬 Validating data consistency...')
    
    # Total charges should generally >= monthly charges except for new customers
    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A = 'TotalCharges',
        column_B = 'MonthlyCharges',
        or_equal = True,
        mostly = .95 # 5% exceptions for new customers
    )
    
    # 6️⃣ Running validation
    print('⚙️ Running complete validation operations...')
    results = ge_df.validate()
    
    # 7️⃣ Results
    failed_expectations = []
    for r in results['results']:
        if not r['success']:
            expectation_type = r['expectation_config']['expectation_type']
            failed_expectations.append(expectation_type)
            
    # 8️⃣ Print validation summary
    total_checks = len(results['results'])
    passed_checks = sum(1 for r in results['results'] if r['success'])
    failed_checks = total_checks - passed_checks
    
    if results["success"]:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")
        
    return results['success'], failed_expectations