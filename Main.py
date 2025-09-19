"""
To run application: streamlit run Main.py --server.port=8502

"""

import streamlit as st
import xgboost as xgb

from cpt_to_soiltype.plotting import plot_predictions
from cpt_to_soiltype.utility import load_data, load_xgb_model


# Streamlit GUI
def main():
    st.title("CPT Prediction")
    st.write(
        "This is a demo of a model that predicts soil type based on Cone Penetration Data (CPT) 🚀"
    )

    with st.expander("About"):
        st.write(
            "The model uses an XGBoost classifier, trained on the Oberhollenzer dataset: [https://doi.org/10.1016/j.dib.2020.106618](https://doi.org/10.1016/j.dib.2020.106618)"
        )
        st.write(
            "The model is built in the applied machine learning course at the Norwegian Geotechnical Institute (NGI): [https://www.ngi.no/](https://www.ngi.no/)"
        )
        st.write(
            "For more information about geotechnical data and applied machine learning, check out this NGI course: [Introduction to Applied Machine Learning - Using Geotechnical Data](https://www.ngi.no/en/events/ngi-code-academy/applied-machine-learning-using-geotechnical-data/)"
        )

    # include an expander that describe the dataformat for the csv file to upload
    with st.expander("Data Format"):
        st.write("The uploaded CSV file should contain the following columns:")
        st.write(
            "- Depth (m)\n"
            "- qc (MPa)\n"
            "- fs (kPa)\n"
            "- Rf (%)\n"
            "- σ,v (kPa)\n"
            "- u0 (kPa)\n"
            "- σ',v (kPa)\n"
            "- Qtn (-)\n"
            "- Fr (%)"
        )
        st.write(
            "Use comma as the delimiter and ensure the column names are exactly as shown above."
        )
        st.write(
            "If the csv file has a column named 'Oberhollenzer_classes', it can be used as the true labels for comparison."
        )
        st.write(
            "The model will predict the 'Oberhollenzer_classes' column based on these features."
        )

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    # Load the XGBoost model
    model_path = "models/xgb_model.json"
    model = load_xgb_model(model_path)

    if uploaded_file is not None:
        # Load data from the uploaded CSV
        df, features, labels = load_data(uploaded_file)

        # Add a checkbox to optionally show true labels
        show_labels = st.checkbox("Show True Labels (if available)")

        # Convert features to DMatrix (excluding the labels)
        dnew = xgb.DMatrix(data=features)

        # Perform predictions
        y_pred_loaded = model.predict(dnew)

        # Plot the predictions
        plot_predictions(df, y_pred_loaded, labels, show_labels)


if __name__ == "__main__":
    main()
