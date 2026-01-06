import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the Model and Encoders
# @st.cache_resource keeps the model in memory so it doesn't reload every time you click a button
@st.cache_resource
def load_assets():
    try:
        with open('cricket_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return model, encoders
    except FileNotFoundError:
        st.error("Error: .pkl files not found. Make sure 'cricket_model.pkl' and 'encoders.pkl' are in the same folder as app.py")
        return None, None

model, encoders = load_assets()

if model and encoders:
    # 2. Extract options for dropdowns from the encoders
    # This ensures the dropdowns only show teams/venues the model actually knows
    team_list = encoders['Team1 Name'].classes_
    venue_list = encoders['Match Venue (Country)'].classes_

    # 3. Build the Interface
    st.title("🏏 T20 International Match Predictor")
    st.markdown("Use Machine Learning to predict the winner of a T20 match.")

    # Create two columns for Team Selection
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Team 1")
        team1 = st.selectbox("Select Team 1", team_list, index=0)
        # Stats for Team 1
        t1_wins = st.number_input(f"{team1} - Wins in Last 10 Matches", min_value=0, max_value=10, value=5)
        t1_runs = st.number_input(f"{team1} - Avg Runs in Last 10 Matches", min_value=0, max_value=300, value=140)

    with col2:
        st.subheader("Team 2")
        # Default index=1 so it doesn't pick the same team as col1 initially
        team2 = st.selectbox("Select Team 2", team_list, index=1)
        # Stats for Team 2
        t2_wins = st.number_input(f"{team2} - Wins in Last 10 Matches", min_value=0, max_value=10, value=5)
        t2_runs = st.number_input(f"{team2} - Avg Runs in Last 10 Matches", min_value=0, max_value=300, value=140)

    st.markdown("---")
    
    # Match Details
    st.subheader("Match Conditions")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        venue = st.selectbox("Match Venue (Country)", venue_list)
    
    with col4:
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
    
    with col5:
        # Note: We convert to lowercase later to match training data
        toss_choice = st.radio("Toss Decision", ["Bat", "Field"])

    # 4. Prediction Logic
    if st.button("Predict Winner"):
        try:
            # A. Encode the Inputs (Text -> Numbers)
            t1_enc = encoders['Team1 Name'].transform([team1])[0]
            t2_enc = encoders['Team2 Name'].transform([team2])[0]
            venue_enc = encoders['Match Venue (Country)'].transform([venue])[0]
            toss_winner_enc = encoders['Toss Winner'].transform([toss_winner])[0]
            
            # --- ROBUST TOSS DECISION ---
            # Check what the model expects for Toss Choice (bat/field vs bat/bowl)
            toss_classes = encoders['Toss Winner Choice'].classes_
            
            # Normalize user input
            raw_choice = 'bat' if toss_choice == 'Bat' else 'field'
            
            # Smartly map to whatever the encoder has
            if raw_choice in toss_classes:
                final_choice = raw_choice
            elif 'bowl' in toss_classes and raw_choice == 'field':
                final_choice = 'bowl' # Map field -> bowl
            else:
                final_choice = toss_classes[0] # Fallback to first available option
                
            toss_choice_enc = encoders['Toss Winner Choice'].transform([final_choice])[0]

            # --- ROBUST HOME ADVANTAGE ---
            # Logic must match the function used in training exactly
            venue_str = str(venue).strip().lower()
            t1_str = str(team1).strip().lower()
            t2_str = str(team2).strip().lower()

            # 1. Determine the logical relationship
            if venue_str == t1_str:
                home_status = 'Team 1'
            elif venue_str == t2_str:
                home_status = 'Team 2'
            else:
                home_status = 'Nuetral'
            
            # 2. Check what labels the encoder ACTUALLY has
            home_classes = encoders['Home Advantage'].classes_
            
            # 3. Match the logic to the available labels
            if home_status in home_classes:
                final_home_status = home_status
            elif home_status.replace(" ", "") in home_classes: # Check "Team1" vs "Team 1"
                final_home_status = home_status.replace(" ", "")
            else:
                # If 'Team 1' isn't found, it usually means the model treats it as Neutral
                # or the label was dropped. Safe fallback:
                final_home_status = 'Neutral' 
                # Only show warning if it's NOT supposed to be neutral
                if home_status != 'Neutral':
                    st.warning(f"Note: Model doesn't recognize '{home_status}'. Defaulting to Neutral.")

            home_adv_enc = encoders['Home Advantage'].transform([final_home_status])[0]

            # C. Create Feature Array (1 row, 10 columns)
            features = np.array([[
                t1_enc, t2_enc, venue_enc,
                toss_winner_enc, toss_choice_enc, home_adv_enc,
                t1_wins, t1_runs,
                t2_wins, t2_runs
            ]])

            # D. Get Prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]

            # E. Display Output
            st.markdown("---")
            if prediction == 1:
                winner = team1
                win_prob = probability[1]
                loss_prob = probability[0]
            else:
                winner = team2
                win_prob = probability[0]
                loss_prob = probability[1]

            st.success(f"🏆 Prediction: **{winner}** will win!")
            st.info(f"Confidence: **{win_prob:.2%}**")

            # Chart Logic
            final_chart = pd.DataFrame({
                'Team': [team1, team2],
                'Win %': [probability[1], probability[0]]
            })
            
            st.bar_chart(final_chart.set_index('Team'))

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            # Debugging Help
            st.write("--- Debugging Info ---")
            st.write(f"Home Advantage Encoder Classes: {encoders['Home Advantage'].classes_}")
            st.write(f"Toss Choice Encoder Classes: {encoders['Toss Winner Choice'].classes_}")