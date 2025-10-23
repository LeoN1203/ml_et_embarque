import streamlit as st
import joblib
import pandas as pd


def predict_price(size, nb_rooms, garden):
    """Predict the price of a house given its features."""
    model = joblib.load("regression.joblib")
    input_data = pd.DataFrame(
        [[size, nb_rooms, garden]], columns=["size", "nb_rooms", "garden"]
    )
    prediction = model.predict(input_data)
    return prediction[0]


# # Streamlit app
# st.title("House Price Prediction")

# # Create a form for user input
# with st.form("prediction_form"):
#     size = st.number_input("Enter the size of the house (in square feet):", min_value=0.0, step=1.0)
#     bedrooms = st.number_input("Enter the number of bedrooms:", min_value=0, step=1)
#     garden = st.number_input("Does the house have a garden? (1 for Yes, 0 for No):", min_value=0, max_value=1, step=1)

#     # Submit button
#     submit = st.form_submit_button("Predict")

# # If the form is submitted, make a prediction
# if submit:
#     # Prepare the input data for the model
#     input_data = [[size, bedrooms, garden]]

#     # Make a prediction
#     prediction = model.predict(input_data)

#     # Display the result
#     st.write(f"The predicted price of the house is: ${prediction[0]:,.2f}")

# print(predict_price(1000, 1, 1))  # Example call to the function
