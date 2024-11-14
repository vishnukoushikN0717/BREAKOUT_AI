import os
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

def search_web(query):
    """Perform a web search using SerpAPI."""
    url = "https://serpapi.com/search.json"
    params = {
        "q": query,
        "api_key": SERP_API_KEY,
        "num": 3  # Retrieve up to 3 search results
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get("organic_results", [])
    else:
        st.error("Web search failed.")
        return []

def process_with_llm(prompt):
    """Process the search results with OpenAI GPT to extract specific information."""
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.error("LLM processing failed.")
        return str(e)

def main():
    st.title("AI Agent for Information Retrieval")

    # File Upload
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.write(df.head())

        # Column Selection
        column = st.selectbox("Select the column to query", df.columns)

        # Query Input
        custom_prompt = st.text_input("Enter your prompt (use {entity} as a placeholder)",
                                     "Get the email address of {entity}")

        if st.button("Run Search and Extraction"):
            results = []
            for entity in df[column].dropna():
                query = custom_prompt.replace("{entity}", entity)
                st.write(f"Searching for: {query}")

                # Perform Web Search
                search_results = search_web(query)
                search_text = " ".join([result['snippet'] for result in search_results])

                # Process with LLM
                llm_prompt = f"Extract the most relevant information for {entity} from the following search results:\n{search_text}"
                extracted_info = process_with_llm(llm_prompt)

                # Store Results
                results.append({"Entity": entity, "Extracted Info": extracted_info})

            # Display Results
            result_df = pd.DataFrame(results)
            st.write("Results:")
            st.write(result_df)

            # Download Results
            csv = result_df.to_csv(index=False)
            st.download_button(label="Download Results as CSV", data=csv, file_name="extracted_info.csv", mime="text/csv")

if __name__ == "__main__":
    main()