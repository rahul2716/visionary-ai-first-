import streamlit as st
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Simulated Segmentation
# -------------------------------
def segment_image(image):
    """
    Simulated segmentation: white = open/usable land
    """
    resized = cv2.resize(image, (256, 256))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)  # white = usable
    return mask

# -------------------------------
# Estimate Land Area from Pixels
# -------------------------------
def estimate_land_area(segmented_image, pixel_scale=1.0):
    """
    Estimate area in square meters.
    Default: 1 pixel = 1 m² (can adjust based on satellite zoom)
    """
    usable_pixels = np.sum(segmented_image == 255)
    area_m2 = usable_pixels * pixel_scale
    return area_m2

# -------------------------------
# Recommend Infrastructure
# -------------------------------
def suggest_infrastructure(area_m2):
    suggestions = []
    if area_m2 < 100:
        suggestions.append("✅ Borewell or Handpump")
        suggestions.append("✅ Public Toilet")
    elif area_m2 < 500:
        suggestions.append("✅ Water Tank")
        suggestions.append("✅ Small Clinic or Shop")
    elif area_m2 < 2000:
        suggestions.append("✅ Primary School")
        suggestions.append("✅ Open Park or Anganwadi")
    else:
        suggestions.append("✅ Community Hall or Hospital")
        suggestions.append("✅ Road Network or Market Area")
    return suggestions

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Vision2Village", layout="wide")
st.title("🌍 Vision2Village - Smart Infrastructure Suggestion")

uploaded_file = st.file_uploader("Upload a Satellite Image of the Village", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Satellite Image", use_column_width=True)

    # Step 1: Segment the Image
    segmented = segment_image(np.array(image))
    st.subheader("🧠 Segmented Usable Land Area")
    st.image(segmented, caption="White = Usable Land", use_column_width=True)

    # Step 2: Estimate Area
    st.subheader("📏 Estimated Usable Land Area")
    pixel_scale = st.slider("Estimate: How many square meters per pixel?", min_value=0.1, max_value=10.0, value=1.0)
    area_m2 = estimate_land_area(segmented, pixel_scale=pixel_scale)
    st.success(f"Estimated Usable Land Area: **{area_m2:.2f} square meters**")

    # Step 3: Suggest Infrastructure
    st.subheader("🏗️ Infrastructure Suggestions")
    suggestions = suggest_infrastructure(area_m2)
    for s in suggestions:
        st.write(s)

else:
    st.info("Upload a satellite image to begin analysis.")


                                #t    #------------------WITH GOOGLE MAP -----------------#

# import streamlit as st
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image
# import requests
# from io import BytesIO
# import leafmap.foliumap as leafmap

# # -------------------------------
# # Simulated Image Segmentation
# # -------------------------------
# def segment_image(image):
#     resized = cv2.resize(image, (256, 256))
#     gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
#     _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
#     return mask

# # -------------------------------
# # Estimate Area
# # -------------------------------
# def estimate_land_area(segmented_image, pixel_scale=1.0):
#     usable_pixels = np.sum(segmented_image == 255)
#     area_m2 = usable_pixels * pixel_scale
#     return area_m2

# # -------------------------------
# # Suggest Infrastructure
# # -------------------------------
# def suggest_infrastructure(area_m2):
#     suggestions = []
#     if area_m2 < 100:
#         suggestions.append("✅ Borewell or Handpump")
#         suggestions.append("✅ Public Toilet")
#     elif area_m2 < 500:
#         suggestions.append("✅ Water Tank")
#         suggestions.append("✅ Small Clinic or Shop")
#     elif area_m2 < 2000:
#         suggestions.append("✅ Primary School")
#         suggestions.append("✅ Open Park or Anganwadi")
#     else:
#         suggestions.append("✅ Community Hall or Hospital")
#         suggestions.append("✅ Road Network or Market Area")
#     return suggestions

# # -------------------------------
# # Main Streamlit App
# # -------------------------------
# st.set_page_config(layout="wide")
# st.title("🌍 Vision2Village - Smart Infrastructure Planner")

# st.markdown("### 📌 Step 1: Select an Area on the Map")

# m = leafmap.Map(center=[20.5937, 78.9629], zoom=5)
# m.add_draw_control()
# m.to_streamlit(height=500)

# st.markdown("### 📍 Step 2: Enter Coordinates of the Area (from map or manually)")

# lat1 = st.number_input("Top-Left Latitude", value=10.0)
# lon1 = st.number_input("Top-Left Longitude", value=78.0)
# lat2 = st.number_input("Bottom-Right Latitude", value=9.995)
# lon2 = st.number_input("Bottom-Right Longitude", value=78.005)

# GOOGLE_MAPS_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"  # Replace with your API key

# if st.button("📷 Fetch Satellite Image"):
#     center_lat = (lat1 + lat2) / 2
#     center_lon = (lon1 + lon2) / 2

#     # Create Google Maps Static URL
#     url = f"https://maps.googleapis.com/maps/api/staticmap?center={center_lat},{center_lon}&zoom=17&size=640x640&maptype=satellite&key={GOOGLE_MAPS_API_KEY}"

#     response = requests.get(url)
#     if response.status_code == 200:
#         image = Image.open(BytesIO(response.content)).convert("RGB")
#         st.image(image, caption="📡 Satellite Image", use_column_width=True)

#         # Convert to numpy array
#         image_np = np.array(image)

#         # Step 3: Segment image
#         st.markdown("### 🧠 Step 3: AI-based Usable Land Detection")
#         segmented = segment_image(image_np)
#         st.image(segmented, caption="Segmented (White = Usable Land)", use_column_width=True)

#         # Step 4: Estimate Area
#         st.markdown("### 📏 Step 4: Estimated Usable Area")
#         pixel_scale = st.slider("Estimate: square meters per pixel", min_value=0.1, max_value=10.0, value=1.0)
#         area_m2 = estimate_land_area(segmented, pixel_scale)
#         st.success(f"Estimated Usable Land Area: **{area_m2:.2f} m²**")

#         # Step 5: Suggest Infrastructure
#         st.markdown("### 🏗️ Step 5: Recommended Infrastructure")
#         suggestions = suggest_infrastructure(area_m2)
#         for s in suggestions:
#             st.write(s)

#     else:
#         st.error("Failed to fetch image. Check coordinates or API key.")
