import streamlit as st
from fastai import *
from fastai.vision import *
import PIL
import torchvision.transforms as T

path = Path(__file__).parent


@st.cache(allow_output_mutation = True)
def learner(path):
	learn = load_learner(path)
	return learn

learn = learner(path)

def main():
	st.set_option('deprecation.showfileUploaderEncoding', False)
	html_title = """  
	<div style="text-align:center;"> 
		<h1>Mask or Not</h1>
	</div>
	"""
	st.markdown(html_title, unsafe_allow_html = True)
	st.subheader("Detect whether masks are worn properly")
	uploaded_file = st.file_uploader("Choose a image file", type=['png', 'jpg', 'jpeg'])
	if uploaded_file is not None:
		img_pil = PIL.Image.open(uploaded_file)
		img_tensor = T.ToTensor()(img_pil)
		our_image = Image(img_tensor)
		st.image(uploaded_file, width=200)
		pred = learn.predict(our_image)[0]
		st.success(pred)
	footer = """
	<div style="position:fixed; text-align:center; bottom:0px; right:0px; left:0px; background-color:grey" markdown="1">
		<h3>Built with Streamlit using fast.ai</h3>
	</div>
	"""
	st.markdown(footer, unsafe_allow_html=True)
	


		



if __name__=='__main__':
    main()

