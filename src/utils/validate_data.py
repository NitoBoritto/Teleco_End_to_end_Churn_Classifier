import great_expectations as gx
from typing import Tuple, List

def validate_data(df) -> Tuple[bool, list[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    
    """
    print("🔎 Starting data validation check with Great Expectations...")
    
    context = gx.get_context()
    datasource_name = "my_pandas_datasource"
    asset_name = "telco_churn_asset"
    
    # 1. Get or Add the Data Source
    try:
        datasource = context.data_sources.get(datasource_name)
    except KeyError:
        datasource = context.data_sources.add_pandas(name=datasource_name)
    
    # 2. Get or Add the Asset
    try:
        asset = datasource.get_asset(asset_name)
    except (LookupError, KeyError):
        print(f"✨ Creating new data asset: {asset_name}")
        asset = datasource.add_dataframe_asset(name=asset_name)

    # 3. Create the Validator (The 2026 Fluent Way)
    # We build an empty batch request and pass the dataframe to the validator
    batch_request = asset.build_batch_request(options = {'dataframe': df}) 
    validator = context.get_validator(batch_request=batch_request)
    
    # 1️⃣ Schema validation
    print("💬 Validating schema and required columns...")
    
    # Customer identifier must exist
    validator.expect_column_to_exist("customerID")
    validator.expect_column_values_to_not_be_null("customerID")
    
    # Core demographic features
    validator.expect_column_to_exist("gender")
    validator.expect_column_to_exist("Partner")
    validator.expect_column_to_exist("Dependents")
    
    # Service features
    validator.expect_column_to_exist("PhoneService")
    validator.expect_column_to_exist("InternetService")
    validator.expect_column_to_exist("Contract")
    
    # Financial features
    validator.expect_column_to_exist("tenure")
    validator.expect_column_to_exist("MonthlyCharges")
    validator.expect_column_to_exist("TotalCharges")
    
    # 2️⃣ Business logic validation
    print("💬 Validating business logic constraints")
    
    # Gender data integrity
    validator.expect_column_values_to_be_in_set('gender', ['Male', 'Female'])
    
    # Yes/No fields data integrity
    validator.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    validator.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    validator.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])
    
    # Contract types must be valid
    validator.expect_column_values_to_be_in_set(
        'Contract',
        ['Month-to-month', 'One year', 'Two year']
    )
    # Internet service types
    validator.expect_column_values_to_be_in_set(
        'InternetService',
        ['DSL', 'Fiber optic', 'No']
    )
    
    # 3️⃣ Numeric Validations
    print('💬 Validating numeric ranges and business constraints...')
    
    # Tenure must be non-negative
    validator.expect_column_values_to_be_between('tenure', min_value=0)
    
    # Monthly charges must be positive
    validator.expect_column_values_to_be_between('MonthlyCharges', min_value=0)
    
    # 4️⃣ Statistical Validation
    print('💬 Validating statistical properties...')
    
    # Tenure should be reasonable (max ~10 years = 120 months for telecom)
    validator.expect_column_values_to_be_between('tenure', min_value=0, max_value=120)
    
    # Monthly charges should be within reasonable business range
    validator.expect_column_values_to_be_between('MonthlyCharges', min_value=0, max_value=200)
    
    # No missing values in critical numeric features
    validator.expect_column_values_to_not_be_null('tenure')
    validator.expect_column_values_to_not_be_null('MonthlyCharges')
    
    # 5️⃣ Data Consistency
    print('💬 Validating data consistency...')
    
    # Total charges should generally >= monthly charges except for new customers
    validator.expect_column_pair_values_A_to_be_greater_than_B(
        column_A = 'TotalCharges',
        column_B = 'MonthlyCharges',
        or_equal = True,
        mostly = 0.95  # 5% exceptions for new customers
    )
    
    # 6️⃣ Running validation
    print('⚙️ Running complete validation operations...')
    results = validator.validate()
    
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
