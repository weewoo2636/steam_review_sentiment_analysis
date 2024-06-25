# import libraries
import streamlit as st

# import modules
import eda
import prediction


def main():
    PAGES = {
        "EDA": eda,
        "Prediction": prediction
    }

    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page.app()

# main idiom
if __name__ == "__main__":
    main()