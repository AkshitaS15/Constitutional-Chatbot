import streamlit as st
from llama_index.core import QueryBundle
from app import custom_query_engine  # Ensure this points to the right import

# Streamlit App

st.title("Constitutional Chatbot....ask  anything related to constitution")

# Input box for the query
query = st.text_input("Enter your query:", "")

# Button to submit the query
if st.button("Submit"):
    if query:
        st.write("Processing your query...")
        try:
            # Pass the query to the custom_query_engine
            response = custom_query_engine.query(query)
            
            # Display the response
            st.subheader("Response:")
            st.write(str(response))
        except Exception as e:
            st.error(f"Error occurred: {str(e)}")
    else:
        st.warning("Please enter a query before submitting.")

