# import lib
import streamlit as st
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import requests
from bs4 import BeautifulSoup
# import io

labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango', 20: 'onion', 21: 'orange',
          22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish',
          29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = [ 'banana', 'apple', 'pear', 'grapes', 'orange', 'kiwi', 'watermelon', 'pomegranate', 'pineapple', 'mango']

vegetables = ['cucumber', 'carrot', 'capsicum', 'onion', 'potato', 'lemon', 'tomato', 'raddish', 'beetroot', 'cabbage', 'lettuce', 'spinach', 'soy bean', 'cauliflower', 'bell pepper', 'chilli pepper', 'turnip', 'corn', 'sweetcorn', 'sweet potato', 'paprika', 'jalepeño', 'ginger', 'garlic', 'peas', 'eggplant']



model = load_model("FruitModel.h5")


def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in a ' + prediction
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = requests.get(url, headers=headers).text
        scrap = BeautifulSoup(req, 'html.parser')

        # Debugging: Print the HTML to check if it's what you expect
        # print(scrap.prettify())

        # calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd")
        calories = scrap.find("div", class_="Z0LcW.an_fna")
        if calories:
            return calories.text
        else:
            st.error("Sorry! Element not found.")
            return None

    except Exception as e:
        st.error("Sorry! Calories not found.")
        print(e)  # Debugging: Print exception to see what went wrong


def process_image(image_stream):
    image = Image.open(image_stream).resize((224, 224))
    img_array = img_to_array(image)
    img_array = img_array / 255
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=-1)[0]
    return labels[predicted_class].capitalize()


def main():
    st.title("Fruit and Vegetable Calories Predictor")
    st.subheader("This is a simple web app to predict calories of fruits and vegetables")
    st.write("Please upload an image of a fruit or vegetable")

    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if image_file:
        img = Image.open(image_file).resize((250, 250))
        st.image(img, use_column_width=True)

        result = process_image(image_file)
        category = "Vegetables" if result.lower() in vegetables else "Fruits"

        st.info(f"Category : {category}")
        st.success(f"The predicted class is: {result}")

        calories = fetch_calories(result)
        if calories:
            st.warning(f"Calories : {calories}")


if __name__ == "__main__":
    main()



