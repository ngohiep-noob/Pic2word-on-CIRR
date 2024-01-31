import streamlit as st
from PIL import Image
from itertools import cycle
from app.main import retrieve, imgs_from_indices


def transform_display_image(image):
    pil_image = Image.open(image)
    processed_image = pil_image.resize((128, 128))
    return processed_image


def process_image(uploaded_image):
    # Open the uploaded image using PIL
    pil_image = Image.open(uploaded_image)

    # Process the image (e.g., resize)
    processed_image = pil_image.resize((224, 224))

    return processed_image


st.set_page_config(
    page_title="Pic2Word",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <h1 style='text-align: center; color: black; padding-top:10px'>Composed image retrieval on
        <a href="https://cirr.cecs.anu.edu.au/">CIRR</a> dataset
    </h1>
    """,
    unsafe_allow_html=True,
)

input_section, output_section = st.columns([1, 2], gap="medium")

with input_section:
    # Add a textbox
    caption = st.text_input("Enter textual description:")

    # Add an image uploader
    uploaded_image = st.file_uploader(
        "Choose a reference image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_image is not None:
        # Display uploaded image
        st.image(
            transform_display_image(uploaded_image),
            caption="Uploaded image",
        )

    # Add a slider for grid size
    n_images = st.slider("Select number of output images", 5, 100, 5)

    submitted = st.button("Search")

with output_section:
    if submitted and uploaded_image is not None:
        st.markdown(
            f"""
            <h3 style='text-align: center; color: black; padding-top:10px'>Top {n_images} results</h3>
            """,
            unsafe_allow_html=True,
        )
        cols = cycle(st.columns(4))

        # Retrieve images
        ref_img = Image.open(uploaded_image)
        indices, distances = retrieve(ref_img, caption, top_k=n_images)

        # Display retrieved images in a grid
        imgs = imgs_from_indices(indices)

        count = 0
        for i, col in zip(imgs, cols):
            col.image(
                i[0].resize((150, 150)), caption=f"{i[1]}({distances[count]:.2f})"
            )
            count += 1
