import streamlit as st
from fastai import *
from fastai.vision import *
import PIL
import torchvision.transforms as T

path = Path(__file__).parent

#cached to avoid reloading the model
@st.cache(allow_output_mutation = True)
def learner(path):						#load the learner from 'export.pkl'
	learn = load_learner(path)	
	return learn



def main():
	st.set_option('deprecation.showfileUploaderEncoding', False)
	html_title = """  
	<div style="text-align:center;"> 
		<h1>Mask or Not</h1>
	</div>
	"""
	st.markdown(html_title, unsafe_allow_html = True)
	st.subheader("Detect whether masks are worn properly")
	learn = learner(path)
	uploaded_file = st.file_uploader("Choose a image file", type=['png', 'jpg', 'jpeg'])
	if uploaded_file is not None:
		st.image(uploaded_file, width=200)
		our_image = open_image(uploaded_file)	#converting to an Image object		
		pred = learn.predict(our_image)[0]	
		st.success(pred)
	footer = """
	<div style="position:fixed; text-align:center; bottom:0px; right:0px; left:0px; background-color:grey" markdown="1">
		<h4>Built with Streamlit using <a href= "https://www.fast.ai">fast.ai</a></h4>
		<p><a href="https://github.com/apzl/mask-or-not">Github</a> | &copyapsal</p>
	</div>
	"""
	st.markdown(footer, unsafe_allow_html=True)
	


		



if __name__=='__main__':
    main()

