import streamlit as st

st.set_page_config(page_title="App Recovery")
st.title("🚨 App Recovery Mode")

st.info("The server hit a memory limit. Click the button below to wipe the server's memory and restore the app.")

if st.button("🔥 WIPE SERVER MEMORY & RESTORE", use_container_width=True, type="primary"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Server memory wiped! I am now restoring your app code. Please wait 10 seconds and refresh.")
    # This button will trigger a rerun which we will use in the NEXT step to restore the code.
