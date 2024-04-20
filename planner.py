import pandas as pd 
import streamlit as st
from datetime import datetime
import os 


try:
    tasks = pd.read_csv("educational_tasks.csv")
except FileNotFoundError:
    tasks = pd.DataFrame(columns=["task", "deadline", "completed"])


st.header("Your Study Planner")

uploaded_file = st.file_uploader("Upload your chapter")
directory = "./uploaded_chapters"
if not os.path.exists(directory):
    os.makedirs(directory)
if uploaded_file is not None:
    file_path = f"./uploaded_chapters/{uploaded_file.name}"  
    try:
        with open(file_path, "wb") as f:  
            f.write(uploaded_file.getbuffer()) 
        st.success("Your file was uploaded and saved locally! You can start learning soon!")

    except Exception as e:
        st.error(f"Error saving your file locally: {e}")

with st.form("Planner"):
    task_text = st.text_input("Add a new study task:")
    task_deadline = st.date_input("Task Deadline", min_value=datetime.today())
    submit_task = st.form_submit_button("Add Task")
    if submit_task and task_text:
        new_task = pd.DataFrame([{"task": task_text, "deadline": task_deadline, "completed": False}])
        tasks = pd.concat([tasks, new_task], ignore_index=True)
        tasks.to_csv("educational_tasks.csv", index=False)
        st.success("Task added successfully!")


st.subheader("Upcoming Study Tasks")
for index, row in tasks.iterrows():
    if not row['completed']: 
        if st.checkbox(f"Mark '{row['task']}' as completed", key=str(index)):
            tasks.at[index, 'completed'] = True
            tasks.to_csv("educational_tasks.csv", index=False)
            st.experimental_rerun()

st.write(tasks)