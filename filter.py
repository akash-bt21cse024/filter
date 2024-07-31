import streamlit as st
from PIL import Image
from opencv-python import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="FILTER IMAGE",layout='centered')

st.title("Image Upload")

    # Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

def numpy_array_to_bytes(np_array, format='PNG'):
    # Convert the NumPy array to a PIL image
    image = Image.fromarray(np_array)
    
    # Save the PIL image to a bytes buffer
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    
    return img_byte_arr.getvalue()
def download(img,count):
    image_bytes = numpy_array_to_bytes(img, format='PNG')
    count = np.random.randint(1, 1000)
    st.download_button(
        label="Download Image",
        data=image_bytes,
        file_name="downloaded_image"+str(count) +".png",
        mime="image/png"
    )
    
if uploaded_file is not None:
        # Open the uploaded image file
    img = Image.open(uploaded_file)
    
    img=np.array(img)   
    
    
    
    
    st.image(img, caption="original Image",width=400,)
    
    tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(['BLUR','BRIGHTNESS','WARM','COOL','FACE BLUR','SKETCH'])
    with tab1:
        num_points = st.slider("Number of blur points", min_value=0, max_value=10, value=0)
        
        if num_points!=0:
           blr_3 = cv.blur(img,(4*num_points,4*num_points))
        else:
           blr_3=img 
        st.image(blr_3, caption="blur Image",width=400)
        count = np.random.randint(1, 1000)
        download(blr_3,count)
        
    with tab6:
        num_points = st.slider("reduce sketch with number", min_value=0, max_value=5, value=2)
        
        blr=cv.blur(img,(6,6))
        edge_1 = cv.Canny(blr, 25*num_points,30*num_points)
       
        
        st.image(edge_1, caption="edge 1",width=400)
        count = np.random.randint(1000, 2000)
        download(edge_1,count)
    with tab2:
        num_points = st.slider("Number of brightness points", min_value=-10, max_value=10, value=0)
        pixels = float(10*num_points)
        img_1 = img + pixels
        img_1[img_1 <  0 ] = 0
        img_1[img_1 > 255] = 255
        img_1 = img_1.astype(np.uint8)
        st.image(img_1, caption="edge 1",width=400)
        count = np.random.randint(2000, 3000)
        download(img_1,count)
        
    with tab4:
        num_points = st.slider("Number of cool points", min_value=0, max_value=9, value=0)
        blue = [0,0,255]
        h=img.shape[0]
        w=img.shape[1]
        
        background = []

        for i in range(h):
            temp = []
            for j in range(w):
                temp.append(blue)
            background.append(temp)
            
        background = np.array(background).astype(np.uint8)
        merged = cv.addWeighted(img, .1*(10-num_points), background, .1*num_points, 0)
        st.image(merged, caption="edge 1",width=400)
        count = np.random.randint(3000, 4000)
        download(merged,count)
        
    with tab3:
        num_points = st.slider("Number of warm points", min_value=0, max_value=8, value=0)
        yellow = [255,250,205]
        h=img.shape[0]
        w=img.shape[1]
        
        background = []

        for i in range(h):
            temp = []
            for j in range(w):
                temp.append(yellow)
            background.append(temp)
            
        background = np.array(background).astype(np.uint8)
        merge = cv.addWeighted(img, .1*(10-num_points), background, .1*num_points, 0)
        st.image(merge, caption="edge 1",width=400)
        count = np.random.randint(4000, 5000)
        download(merge,count)
    with tab5:
        classifier = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if classifier.empty():
          raise IOError("Failed to load cascade classifier. Check the file path.")
        if img is None:
          raise IOError("Failed to load image. Check the image path.")

        
        faces = classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        
        for f in faces:
         if f[-1] == max(faces[:,-1]):
            break

        if (len(faces) >= 1):
            x = f[0] 
            y = f[1] 
            w = f[2]
            h = f[3]

#         cv.rectangle(img, (x,y),(x+w,y+h) , (0,180,0), 2)   
        num_points = st.slider("Number of face blur points", min_value=1, max_value=10, value=1)
        
        face = img[y:y+h, x:x+w]                 # Getting the Face Area from Video Feed
        face = cv.blur(face, (4*num_points,4*num_points))            # Applying Blur on the Face
        img[y:y+h, x:x+w] = face 
        st.image(img, caption="edge 1",width=400)# Apply Blured Face on Video Feed
        count = np.random.randint(5000, 6000)
        download(img,count)
        
        
        

