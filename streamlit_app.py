import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Text-to-Image AI Generator")

# Initialize Hugging Face client
API_KEY = st.secrets.get("HF_API_KEY") 
client = InferenceClient("black-forest-labs/FLUX.1-dev", token=API_KEY)


# Streamlit app
st.title("Text-to-Image AI Generator")
st.markdown("""
This AI chat app uses the <a href="https://huggingface.co/black-forest-labs/FLUX.1-dev" target="_blank">**FLUX.1-dev**</a> model. FLUX.1 [dev] is a 12-billion-parameter rectified flow transformer capable of generating images from text descriptions.
""", unsafe_allow_html=True)


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 



st.subheader("Enter a text prompt to generate an image")

# Input text prompt
query = st.text_input("Enter your prompt:", "")

if st.button("Generate Image"):
    if query.strip():
        with st.spinner("Generating image..."):
            try:
                # Generate image
                image = client.text_to_image(query)
                
                # Display image
                if image:
                    st.image(image, caption="Generated Image", use_container_width=True)
                    # Option to download the image
                    buf = BytesIO()
                    image.save(buf, format="PNG")
                    buf.seek(0)
                    st.download_button(
                        label="Download Image",
                        data=buf,
                        file_name="generated_image.png",
                        mime="image/png",
                    )
                else:
                    st.error("Failed to generate image. Please try again.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid prompt.")
