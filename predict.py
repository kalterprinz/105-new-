import streamlit as st
import pandas as pd
import joblib

# Streamlit App Title
st.title("Make Predictions using a Trained Model")

# Upload a trained model
model_file = st.file_uploader("Upload your trained model (.pkl file)", type=["pkl"])

# Ensure model is uploaded before proceeding
if model_file is not None:
    # Load the model
    model = joblib.load(model_file)
    st.success("Model successfully loaded!")

    # Check if model has feature names
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        # If feature names are not available, prompt the user to input them
        feature_names = st.text_area("Enter the feature names separated by commas", "").split(",")
        feature_names = [f.strip() for f in feature_names]  # Ensure no trailing spaces

    # Optionally display target labels if they exist in the model
    if hasattr(model, 'classes_'):
        st.write("### Target Labels:")
        st.write(model.classes_)

    # Input the feature values for prediction
    st.write("### Input the feature values for prediction:")
    input_data = {}

    for feature_name in feature_names:
        if feature_name:  # Ensure no empty feature names
            feature_value = st.text_input(f"Enter value for {feature_name}")
            try:
                input_data[feature_name] = float(feature_value) if feature_value else None
            except ValueError:
                st.error(f"Please enter a valid numeric value for {feature_name}")

    # Validate that all feature inputs have been provided
    valid_input = all(v is not None for v in input_data.values())

    # Convert the input to a DataFrame if valid
    if valid_input:
        input_df = pd.DataFrame([input_data])
        st.write("### Input Data Preview:")
        st.write(input_df)

        # Perform the prediction
        if st.button("Predict"):
            try:
                # Ensure the input DataFrame columns match the model's expected feature names
                input_df = input_df[model.feature_names_in_]
                
                prediction = model.predict(input_df)
                st.write(f"### Prediction: {prediction[0]}")
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    else:
        st.warning("Please enter valid input values for all features.")
else:
    st.info("Please upload a trained model to proceed.")
