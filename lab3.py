from itertools import islice
import time
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid, train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, Perceptron, Lasso, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score, roc_curve
import joblib
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Step navigation
steps = ["Upload Data", "Model Training", "Model Comparison", "Model Usage"]
step = st.sidebar.radio("Steps", steps)

# Session state for carrying data between steps
if "data" not in st.session_state:
    st.session_state.data = None
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_train" not in st.session_state:
    st.session_state.y_train = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None

# Step 1: Upload Data
if step == "Upload Data":
    st.title("Step 1: Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data
        st.write("Preview of the dataset:")
        st.dataframe(data.head())
    else:
        st.warning("Please upload a CSV file to proceed.")

# Step 2: Model Training
elif step == "Model Training":
    st.title("Step 2: Model Training")

    if st.session_state.data is None:
        st.warning("Please upload a dataset first in Step 1.")
    else:
        data = st.session_state.data
        st.write("Dataset loaded successfully!")

        # Select mode: Classification or Regression
        mode = st.radio("Select the task type:", ["Classification", "Regression"])

        target = st.selectbox("Select the target variable:", data.columns)
        features = st.multiselect("Select feature variables:", data.columns, default=[col for col in data.columns if col != target])

        if features and target:
            X = data[features]
            y = data[target]

            # Handle missing values
            st.subheader("Handle Missing Data")
            handle_missing = st.radio(
                "Choose how to handle missing values:",
                ["Impute with Mean", "Drop Rows with Missing Values"]
            )
            if handle_missing == "Impute with Mean":
                X.fillna(X.mean(), inplace=True)
                y.fillna(y.mean(), inplace=True)
            elif handle_missing == "Drop Rows with Missing Values":
                X.dropna(inplace=True)
                y = y.loc[X.index]  # Ensure the target matches the filtered features

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            st.session_state.X_train, st.session_state.X_test = X_train, X_test
            st.session_state.y_train, st.session_state.y_test = y_train, y_test

            st.success("Data split into training and test sets!")
            st.write(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

            # Select models based on mode
            if mode == "Classification":
                models = {
                    "Decision Tree (CART)": DecisionTreeClassifier(),
                    "Naive Bayes": GaussianNB(),
                    "Gradient Boosting Machines (AdaBoost)": AdaBoostClassifier(),
                    "K-Nearest Neighbors (K-NN)": KNeighborsClassifier(),
                    "Logistic Regression": LogisticRegression(),
                    "Multi-Layer Perceptron (MLP)": MLPClassifier(),
                    "Perceptron": Perceptron(),
                    "Random Forest": RandomForestClassifier(),
                    "Support Vector Machines (SVM)": SVC(probability=True),
                }
                metric_fn = accuracy_score
                metric_name = "Accuracy"
            elif mode == "Regression":
                models = {
                    "Decision Tree": DecisionTreeRegressor(),
                    "Naive Bayes (Gaussian)": GaussianNB(),  
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "K-Nearest Neighbors": KNeighborsRegressor(),
                    "Linear Regression": LinearRegression(),
                    "Random Forest (Regressor)": RandomForestRegressor(),
                    "Support Vector Regression (SVR)": SVR(),
                    "Multi-Layer Perceptron (MLP Regressor)": MLPRegressor(),
                    "Elastic Net": ElasticNet(),
                    "Lasso Regression": Lasso(),
                    "Ridge Regression": Ridge(),
                }
                metric_fn = mean_absolute_error
                metric_name = "Mean Absolute Error"

            # Train models
            if st.button("Train Models"):
                
                X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, test_size=0.8, random_state=42)

                results = []
                total_models = len(models)
                progress_bar = st.progress(0)
                progress_text = st.empty()

                for idx, (name, model) in enumerate(models.items()):
                    progress_text.text(f"Training {name} ({idx + 1}/{total_models})...")

                    # Train the model
                    start_time = time.time()
                    model.fit(X_train_sample, y_train_sample)
                    y_pred = model.predict(X_test)
                    metric_value = metric_fn(y_test, y_pred)
                    elapsed_time = time.time() - start_time

                    results.append({
                        "Model": name,
                        metric_name: metric_value,
                        "Time (s)": elapsed_time,
                    })

                    progress_bar.progress((idx + 1) / total_models)

                progress_text.text("Training complete!")
                results_df = pd.DataFrame(results)
                st.table(results_df)

                # Plot results
                st.subheader(f"Model {metric_name} Comparison")
                fig, ax = plt.subplots()
                ax.barh(results_df["Model"], results_df[metric_name], color="skyblue")
                ax.set_xlabel(metric_name)
                ax.set_ylabel("Model")
                ax.set_title(f"{metric_name} of Each Algorithm")
                st.pyplot(fig)

                # Save best model
                if mode == "Classification":
                    best_model_name = results_df.loc[results_df[metric_name].idxmax(), "Model"]
                elif mode == "Regression":
                    best_model_name = results_df.loc[results_df[metric_name].idxmin(), "Model"]
                st.session_state.trained_model = models[best_model_name]
                st.success(f"Best model: {best_model_name}")
# Step 3: Model Comparison and Hyperparameter Tuning
elif step == "Model Comparison":
    st.title("Step 3: Model Comparison and Hyperparameter Tuning")

    if "trained_model" not in st.session_state or st.session_state.trained_model is None:
        st.warning("Please train models in Step 2 first.")
    else:
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train

        # Subset the training data for faster tuning (e.g., 20% of the original training data)
        from sklearn.model_selection import train_test_split
        X_sample, _, y_sample, _ = train_test_split(X_train, y_train, test_size=0.8, random_state=42)

        # Dropdown to select the model for hyperparameter tuning
        model_options = {
            "Support Vector Machines (SVM)": SVC(),
            "Random Forest (Classifier)": RandomForestClassifier(),
            "Gradient Boosting Machines": GradientBoostingClassifier(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree (Classifier)": DecisionTreeClassifier(),
        }
        selected_model_name = st.selectbox("Select a Model for Hyperparameter Tuning", model_options.keys())
        selected_model = model_options[selected_model_name]

        # Define hyperparameter grids for supported models
        param_grids = {
            "Support Vector Machines (SVM)": {
                "kernel": ["linear", "rbf", "poly", "sigmoid"],
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"]
            },
            "Random Forest (Classifier)": {
                "n_estimators": [10, 50, 100],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
            },
            "Gradient Boosting Machines": {
                "n_estimators": [50, 100],
                "learning_rate": [0.1, 0.2],
                "max_depth": [3, 5],
            },
            "K-Nearest Neighbors": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
            },
            "Decision Tree (Classifier)": {
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
            }
        }

        # If the selected model has hyperparameters defined, proceed with tuning
        if selected_model_name in param_grids:
            param_grid = param_grids[selected_model_name]

            if st.button("Run Hyperparameter Tuning"):
                st.write(f"Tuning hyperparameters for {selected_model_name} using a sample subset... This may take a while.")

                param_combinations = list(islice(ParameterGrid(param_grid), 4))

                # Initialize progress bar and text
                progress_bar = st.progress(0)
                progress_text = st.empty()

                best_score = -float("inf")
                best_params = None
                results = []

                for idx, params in enumerate(param_combinations):
                    # Update progress
                    progress = (idx + 1) / len(param_combinations)
                    progress_bar.progress(progress)
                    progress_text.text(f"Evaluating {idx + 1}/{len(param_combinations)} combinations...")

                    # Set parameters and fit the model using the sample data
                    model = selected_model.set_params(**params)
                    model.fit(X_sample, y_sample)  # Use the subset for fitting

                    # Evaluate the model
                    score = model.score(X_sample, y_sample)  # Replace with desired metric
                    results.append({"Hyperparameters": params, "Accuracy": score})

                    # Track the best model
                    if score > best_score:
                        best_score = score
                        best_params = params

                # Convert results to DataFrame
                results_df = pd.DataFrame(results)
                results_df["Accuracy"] = results_df["Accuracy"].apply(lambda x: f"{x * 100:.2f}%")

                progress_text.text("Hyperparameter tuning complete!")
                progress_bar.progress(1.0)

                # Display results and best hyperparameters
                st.subheader("Hyperparameter Tuning Results")
                st.write(results_df)  # Display all 4 configurations
                st.success(f"Best Hyperparameters: {best_params}")
                st.write(f"Best Accuracy: {best_score * 100:.2f}%")

                # Save the best model using the full dataset
                st.session_state.trained_model = selected_model.set_params(**best_params)
                st.session_state.trained_model.fit(X_train, y_train)  # Retrain on the full dataset
                st.session_state.selected_model_name = selected_model_name
                st.success(f"The best-tuned {selected_model_name} model has been saved for use in Step 4!")

        else:
            st.error(f"Hyperparameter tuning is not yet implemented for {selected_model_name}.")



# Step 4: Model Usage
elif step == "Model Usage":
    st.title("Step 4: Model Usage")

    if st.session_state.trained_model is None:
        st.warning("Please train a model in Step 2 first.")
    else:
        model = st.session_state.trained_model
        X_test = st.session_state.get("X_test", None)
        y_test = st.session_state.get("y_test", None)

        if X_test is None or y_test is None:
            st.error("Test data not found. Please complete Steps 1 and 2 first.")
        else:
            st.subheader("Predictions and Evaluation")
            try:
                # Make predictions
                y_pred = model.predict(X_test)

                # Display Predictions
                st.subheader("Predictions on Test Data")
                prediction_df = pd.DataFrame({
                    "Actual": y_test.values if hasattr(y_test, "values") else y_test,
                    "Predicted": y_pred
                })
                st.write(prediction_df.head())

                # Attempt Classification Metrics
                try:
                    st.subheader("Classification Metrics")
                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"Accuracy: {accuracy:.2f}")

                    # ROC Curve and AUC (if applicable)
                    if hasattr(model, "predict_proba") and len(set(y_test)) == 2:
                        y_prob = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        roc_auc = roc_auc_score(y_test, y_prob)

                        st.subheader("ROC Curve")
                        fig, ax = plt.subplots()
                        ax.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.2f})")
                        ax.plot([0, 1], [0, 1], 'k--', label="Random Guess")
                        ax.set_xlabel("False Positive Rate")
                        ax.set_ylabel("True Positive Rate")
                        ax.set_title("Receiver Operating Characteristic")
                        ax.legend(loc="lower right")
                        st.pyplot(fig)
                    else:
                        st.warning("ROC-AUC is only applicable for binary classification.")
                except Exception:
                    # st.warning("Switching to regression metrics due to classification metric error.")

                    # Regression Metrics
                    st.subheader("Regression Metrics")
                    mae = mean_absolute_error(y_test, y_pred)
                    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")

                    # Scatter plot of actual vs predicted
                    st.subheader("Actual vs Predicted Plot")
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=y_test, y=y_pred, ax=ax, color="blue", alpha=0.6)
                    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    ax.set_title("Actual vs Predicted")
                    st.pyplot(fig)

                    # Save and download the trained model
                    buffer = io.BytesIO()
                    joblib.dump(model, buffer)
                    buffer.seek(0)
                    st.download_button(
                        label="Download Trained Model",
                        data=buffer,
                        file_name="trained_model.pkl",
                        mime="application/octet-stream",
                    )

                    st.success("You can now use the trained model for predictions.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                
            except Exception as e:
                st.warning("Model evaluation could not be completed.")
