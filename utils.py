def save_temp_file(uploaded_file):
    path = f"temp_{uploaded_file.name}"
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path
