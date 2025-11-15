import base64
import streamlit as st
import pickle
import pandas as pd

# Function to load image as base64 for background
@st.cache_data
def get_img_as_base64(file):
    try:
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error(f"Background image '{file}' not found. Please ensure the file exists.")
        return None

# Load the image for background
img = get_img_as_base64("background.jpeg")  # Adjust the path if necessary
if img:
    # Style for background image
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/png;base64,{img}");
    width: 100%;
    height:100%;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-size: cover;
    }}

    [data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/png;base64,{img}");
    background-position: center; 
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}

    [data-testid="stToolbar"] {{
    right: 2rem;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
else:
    st.warning("Background image could not be loaded.")

# Load the trained model
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file 'pipe.pkl' not found. Please ensure the file exists.")

# List of teams and cities for user selection
teams = ['--- select ---',
         'Sunrisers Hyderabad',
         'Mumbai Indians',
         'Kolkata Knight Riders',
         'Royal Challengers Bangalore',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Bangalore', 'Hyderabad', 'Kolkata', 'Mumbai', 'Visakhapatnam',
          'Indore', 'Durban', 'Chandigarh', 'Delhi', 'Dharamsala',
          'Ahmedabad', 'Chennai', 'Ranchi', 'Nagpur', 'Mohali', 'Pune',
          'Bengaluru', 'Jaipur', 'Port Elizabeth', 'Centurion', 'Raipur',
          'Sharjah', 'Cuttack', 'Johannesburg', 'Cape Town', 'East London',
          'Abu Dhabi', 'Kimberley', 'Bloemfontein']

# Title
st.markdown("""
    # **IPL VICTORY PREDICTOR**            
""")

# Input layout
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select Batting Team', teams)

with col2:
    if batting_team == '--- select ---':
        bowling_team = st.selectbox('Select Bowling Team', teams)
    else:
        filtered_teams = [team for team in teams if team != batting_team]
        bowling_team = st.selectbox('Select Bowling Team', filtered_teams)

selected_city = st.selectbox('Select Venue', cities)

target = st.number_input('Target', min_value=1)

col1, col2, col3 = st.columns(3)

with col1:
    score = st.number_input('Score', min_value=0)

with col2:
    overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)

with col3:
    wickets = st.number_input("Wickets Down", min_value=0, max_value=10)

# Button to predict winning probability
if st.button('Predict Winning Probability'):
    try:
        # Validate inputs
        if batting_team == '--- select ---' or bowling_team == '--- select ---' or selected_city == '--- select ---':
            st.error("Please select both teams and the venue.")
        elif score > target:
            st.error("Score cannot be greater than the target.")
        elif wickets < 0 or wickets > 10:
            st.error("Wickets must be between 0 and 10.")
        elif overs < 0 or overs > 20:
            st.error("Overs should be between 0 and 20.")
        else:
            # Calculate derived features
            runs_left = target - score
            balls_left = 120 - (overs * 6)
            wickets_remaining = 10 - wickets
            crr = score / overs if overs > 0 else 0  # Avoid division by zero
            rrr = runs_left / (balls_left / 6) if balls_left > 0 else 0  # Avoid division by zero

            # Prepare input data for model prediction
            input_data = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [selected_city],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets_remaining': [wickets_remaining],
                'total_runs_x': [target],
                'crr': [crr],
                'rrr': [rrr]
            })

            # Get prediction probabilities
            result = pipe.predict_proba(input_data)

            # Extract probabilities for win and loss
            loss = result[0][0]
            win = result[0][1]

            # Display results
            st.header(f"{batting_team} = {round(win * 100)}%")
            st.header(f"{bowling_team} = {round(loss * 100)}%")

    except Exception as e:
        st.error(f"Some error occurred: {str(e)}")
