import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🌿 كاشف أمراض النباتات")
st.write("ارفع صورة ورقة نبات وهيقولك المرض المحتمل")
st.info("💡 **للحصول على أفضل نتيجة:** يُفضل رفع صورة لورقة واحدة فقط، وأن تكون الخلفية واضحة وبسيطة.")


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_disease_model.keras")
    with open("class_names.txt") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names


model, class_names = load_model()

uploaded_file = st.file_uploader("اختر صورة", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الصورة المرفوعة", use_container_width=True)

    img_resized = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized), axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.subheader(f"النتيجة: {predicted_class}")
    st.write(f"نسبة الثقة: {confidence:.2f}%")
    if confidence < 85:
        st.warning("⚠️ نسبة الثقة غير كافية، يرجى التقاط صورة أقرب وأوضح للبقع.")
