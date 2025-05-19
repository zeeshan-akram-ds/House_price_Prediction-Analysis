import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lime.lime_tabular import LimeTabularExplainer

# Load saved model, scaler, and columns
model = joblib.load("xgb_best_model_housing.pkl")
scaler = joblib.load("house_scaler.pkl")

with open("feature_names.json", "r") as f:
    feature_names = json.load(f)
X_test = joblib.load("X_test_sample.pkl")
y_test = joblib.load("y_test_sample.pkl")

# Define calculated features
CALCULATED_FEATURES = [
    'Total_SF', 'Total_Bathrooms', 'Has_Fireplace', 
    'Has_FinishedBsmt', 'Total_Porch_SF'
]

# feature groups for organized input
FEATURE_GROUPS = {
    "General Property": [
        'Lot Area', 'Lot Frontage', 'Year Built', 'Year Remod/Add', 
        'Yr Sold', 'MS SubClass', 'Overall Qual', 'Overall Cond'
    ],
    "Living Area": [
        'Gr Liv Area', '1st Flr SF', '2nd Flr SF', 'TotRms AbvGrd', 
        'Bedroom AbvGr', 'Kitchen AbvGr'
    ],
    "Basement": [
        'Total Bsmt SF', 'BsmtFin SF 1', 'BsmtFin SF 2', 
        'Bsmt Unf SF', 'Bsmt Full Bath', 'Bsmt Half Bath'
    ],
    "Bathrooms": [
        'Full Bath', 'Half Bath'
    ],
    "Garage": [
        'Garage Area', 'Garage Cars', 'Garage Yr Blt'
    ],
    "Exterior": [
        'Mas Vnr Area', 'Wood Deck SF', 'Open Porch SF', 
        'Enclosed Porch', 'Fireplaces'
    ]
}

# Human-readable labels and tooltips
FEATURE_LABELS = {
    'Lot Area': {'label': 'Lot Size (sq ft)', 'tooltip': 'Total area of the property lot in square feet'},
    'Lot Frontage': {'label': 'Lot Frontage (ft)', 'tooltip': 'Width of the lot facing the street in feet'},
    'Year Built': {'label': 'Year Built', 'tooltip': 'Year the house was constructed'},
    'Year Remod/Add': {'label': 'Year Remodeled', 'tooltip': 'Year of last major remodel or addition'},
    'Yr Sold': {'label': 'Year Sold', 'tooltip': 'Year the house was sold'},
    'MS SubClass': {'label': 'Building Type', 'tooltip': 'Code for building type (e.g., 20 for 1-story)'},
    'Overall Qual': {'label': 'Overall Quality', 'tooltip': 'Rates the overall material and finish (1-10)'},
    'Overall Cond': {'label': 'Overall Condition', 'tooltip': 'Rates the overall condition (1-10)'},
    'Gr Liv Area': {'label': 'Above Ground Living Area (sq ft)', 'tooltip': 'Living area above ground in square feet'},
    '1st Flr SF': {'label': 'First Floor Area (sq ft)', 'tooltip': 'First floor area in square feet'},
    '2nd Flr SF': {'label': 'Second Floor Area (sq ft)', 'tooltip': 'Second floor area in square feet'},
    'TotRms AbvGrd': {'label': 'Total Rooms Above Ground', 'tooltip': 'Total rooms above ground (excl. bathrooms)'},
    'Bedroom AbvGr': {'label': 'Bedrooms Above Ground', 'tooltip': 'Number of bedrooms above ground'},
    'Kitchen AbvGr': {'label': 'Kitchens Above Ground', 'tooltip': 'Number of kitchens above ground'},
    'Total Bsmt SF': {'label': 'Basement Area (sq ft)', 'tooltip': 'Total basement area in square feet'},
    'BsmtFin SF 1': {'label': 'Finished Basement Area (sq ft)', 'tooltip': 'Finished basement area in square feet'},
    'BsmtFin SF 2': {'label': 'Low-Quality Finished Basement (sq ft)', 'tooltip': 'Low-quality finished basement area'},
    'Bsmt Unf SF': {'label': 'Unfinished Basement Area (sq ft)', 'tooltip': 'Unfinished basement area in square feet'},
    'Bsmt Full Bath': {'label': 'Basement Full Bathrooms', 'tooltip': 'Number of full bathrooms in basement'},
    'Bsmt Half Bath': {'label': 'Basement Half Bathrooms', 'tooltip': 'Number of half bathrooms in basement'},
    'Full Bath': {'label': 'Full Bathrooms Above Ground', 'tooltip': 'Number of full bathrooms above ground'},
    'Half Bath': {'label': 'Half Bathrooms Above Ground', 'tooltip': 'Number of half bathrooms above ground'},
    'Garage Area': {'label': 'Garage Area (sq ft)', 'tooltip': 'Garage area in square feet'},
    'Garage Cars': {'label': 'Garage Capacity (cars)', 'tooltip': 'Number of cars the garage can hold'},
    'Garage Yr Blt': {'label': 'Garage Year Built', 'tooltip': 'Year the garage was built'},
    'Mas Vnr Area': {'label': 'Masonry Veneer Area (sq ft)', 'tooltip': 'Masonry veneer area in square feet'},
    'Wood Deck SF': {'label': 'Wood Deck Area (sq ft)', 'tooltip': 'Wood deck area in square feet'},
    'Open Porch SF': {'label': 'Open Porch Area (sq ft)', 'tooltip': 'Open porch area in square feet'},
    'Enclosed Porch': {'label': 'Enclosed Porch Area (sq ft)', 'tooltip': 'Enclosed porch area in square feet'},
    'Fireplaces': {'label': 'Number of Fireplaces', 'tooltip': 'Total number of fireplaces'},
}

st.set_page_config(page_title="Ames Housing Price Predictor", layout="wide")

with st.sidebar:
    st.markdown("### Navigation")
    page = st.selectbox(
        "Choose a section",
        ["Input Features", "Toolbox", "Model Insights"]
    )

st.title("Ames Housing Price Predictor")
st.markdown("""
    Predict the sale price of a house with XGBoost model. 
    Enter details below or explore model insights for a deeper understanding.
""")

# Input Features Section
if page == "Input Features":
    # Initialize session state for inputs
    if 'inputs' not in st.session_state:
        st.session_state.inputs = {col: 0.0 for col in feature_names}

    with st.form("house_features"):
        st.markdown("### House Features")
        for group_name, features in FEATURE_GROUPS.items():
            st.subheader(group_name)
            cols = st.columns(3)
            for i, feature in enumerate(features):
                with cols[i % 3]:
                    label = FEATURE_LABELS[feature]['label']
                    tooltip = FEATURE_LABELS[feature]['tooltip']
                    if "Area" in feature or "SF" in feature or "Lot" in feature:
                        st.session_state.inputs[feature] = st.number_input(
                            label, min_value=0.0, value=100.0, step=10.0, 
                            help=tooltip, key=feature
                        )
                    elif "Yr" in feature or "Year" in feature:
                        st.session_state.inputs[feature] = st.number_input(
                            label, min_value=1800, value=2000, step=1, 
                            help=tooltip, key=feature
                        )
                    elif feature in ['MS SubClass', 'Bedroom AbvGr', 'Kitchen AbvGr', 
                                  'Garage Cars', 'TotRms AbvGrd', 'Fireplaces',
                                  'Bsmt Full Bath', 'Bsmt Half Bath', 'Full Bath', 
                                  'Half Bath', 'Overall Qual', 'Overall Cond']:
                        st.session_state.inputs[feature] = st.number_input(
                            label, min_value=0, value=1, step=1, 
                            help=tooltip, key=feature
                        )

        # Form buttons
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Predict Sale Price")
        with col2:
            clear = st.form_submit_button("Clear All")

        if clear:
            st.session_state.inputs = {col: 0.0 for col in feature_names}
            st.success("All inputs cleared!")
            st.rerun()

        if submitted:
            with st.spinner("Calculating prediction..."):
                input_df = pd.DataFrame([st.session_state.inputs])
                input_df['Total_SF'] = (input_df['Total Bsmt SF'] + 
                                      input_df['1st Flr SF'] + 
                                      input_df['2nd Flr SF'])
                input_df['Total_Bathrooms'] = (input_df['Full Bath'] + 
                                            0.5 * input_df['Half Bath'] +
                                            input_df['Bsmt Full Bath'] + 
                                            0.5 * input_df['Bsmt Half Bath'])
                input_df['Has_Fireplace'] = input_df['Fireplaces'].apply(
                    lambda x: 1 if x > 0 else 0
                )
                input_df['Has_FinishedBsmt'] = input_df['BsmtFin SF 1'].apply(
                    lambda x: 1 if x > 0 else 0
                )
                input_df['Total_Porch_SF'] = (input_df['Open Porch SF'] + 
                                            input_df['Enclosed Porch'])
                input_array = input_df[feature_names].to_numpy().reshape(1, -1)
                input_scaled = scaler.transform(input_array)
                log_prediction = model.predict(input_scaled)[0]
                prediction = np.expm1(log_prediction)
                st.success(f"**Predicted Sale Price:** ${prediction:,.2f}")

                # Store input for LIME
                st.session_state['input_array'] = input_scaled

            with st.expander("Why this prediction? (LIME Explanation)"):
                try:
                    explainer = LimeTabularExplainer(
                        training_data=X_test.values,  
                        feature_names=feature_names,
                        mode="regression",
                        verbose=False,
                        random_state=42,
                        discretize_continuous=True
                    )
                    exp_user = explainer.explain_instance(
                        data_row=input_scaled[0],
                        predict_fn=model.predict,
                        num_features=10
                    )
                    fig_user = exp_user.as_pyplot_figure()
                    st.pyplot(fig_user)
                    plt.close(fig_user)
                except Exception as e:
                    st.error(f"LIME explanation failed: {e}")

# Toolbox Section
elif page == "Toolbox":
    st.markdown("### Toolbox")
    st.markdown("""
        **How to Use This App:**
        - Go to "Input Features" to enter house details.
        - Each input field has a tooltip for guidance.
        - Calculated fields (e.g., Total Square Footage) are computed automatically.
        - Use "Predict Sale Price" to get an estimate or "Clear All" to reset.
        - Visit "Model Insights" for performance metrics and visualizations.

        **Feature Explanations:**
        - **Lot Size**: Total property area in square feet.
        - **Overall Quality**: Material and finish quality (1-10).
        - **Basement Area**: Includes finished and unfinished areas.
        - **Bathrooms**: Full bathrooms include a shower/tub; half do not.

        **Tips:**
        - Use realistic values for Ames, Iowa houses.
        - Enter years between 1800 and 2025.
        - Set fields to 0 for non-applicable features (e.g., no garage).
    """)

# Model Insights Section
else:
    st.markdown("### Model Insights")
    tabs = st.tabs(["Evaluation Metrics", "LIME Explanation", "Feature Importance"])
    
    with tabs[0]:
        st.markdown("#### Model Performance")
        log_preds = model.predict(X_test)
        preds = np.expm1(log_preds)
        actuals = np.expm1(y_test)
        mse = mean_squared_error(actuals, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, preds)
        r2 = r2_score(actuals, preds)
        adj_r2 = 1 - (1 - r2) * (len(actuals) - 1) / (len(actuals) - X_test.shape[1] - 1)
        st.markdown(f"""
            - **RMSE:** ${rmse:,.2f}  
            - **MAE:** ${mae:,.2f}  
            - **R²:** {r2:.4f}  
            - **Adjusted R²:** {adj_r2:.4f}  
        """)
        st.markdown("##### Select a Plot")
        plot_type = st.radio(
            "Choose visualization",
            ["Predicted vs Actual", "Residuals"],
            key="eval_plot"
        )
        if plot_type == "Predicted vs Actual":
            fig1, ax1 = plt.subplots()
            sns.scatterplot(x=actuals, y=preds, ax=ax1)
            ax1.set_title("Predicted vs Actual Sale Prices")
            ax1.set_xlabel("Actual Price ($)")
            ax1.set_ylabel("Predicted Price ($)")
            ax1.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')
            st.pyplot(fig1)
            plt.close(fig1)
        elif plot_type == "Residuals":
            residuals = actuals - preds
            fig2, ax2 = plt.subplots()
            sns.scatterplot(x=preds, y=residuals, ax=ax2)
            ax2.axhline(0, color='red', linestyle='--')
            ax2.set_title("Residuals vs Predicted Values")
            ax2.set_xlabel("Predicted Price ($)")
            ax2.set_ylabel("Residuals ($)")
            st.pyplot(fig2)
            plt.close(fig2)

    with tabs[1]:
        st.markdown("#### LIME Explanation for a Test Instance")
        st.markdown("Select a test house to see which features influenced its predicted price.")
        idx = st.slider("Choose test instance", 0, len(X_test) - 1, 0, key="lime_slider")
        if st.button("Explain Instance with LIME", key="lime_button"):
            with st.spinner("Generating explanation..."):
                explainer = LimeTabularExplainer(
                    training_data=X_test.values,       
                    feature_names=feature_names,
                    mode="regression",
                    discretize_continuous=True,
                    random_state=42
                )
                instance = X_test.iloc[idx].values
                exp = explainer.explain_instance(
                    data_row=instance,
                    predict_fn=model.predict,
                    num_features=10
                )
                fig = exp.as_pyplot_figure()
                st.pyplot(fig)
                plt.close(fig)

    with tabs[2]:
        st.markdown("#### Feature Importance (XGBoost)")
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        st.markdown("##### Select a Plot")
        plot_type = st.radio(
            "Choose visualization",
            ["Feature Importance"],
            key="feat_plot"
        )
        if plot_type == "Feature Importance":
            st.markdown("###### Top 15 Important Features")
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            sns.barplot(
                data=importance_df.head(15), 
                x='Importance', y='Feature', hue='Feature', palette='viridis', ax=ax3, legend=False
            )
            ax3.set_title("Top 15 Features by Importance (XGBoost)")
            st.pyplot(fig3)
            plt.close(fig3)
        st.markdown("##### Full Feature Importance Table")
        st.dataframe(importance_df.reset_index(drop=True), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Developed by **Zeeshan Akram** | Built with Streamlit | © 2025")