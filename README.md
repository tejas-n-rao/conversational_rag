#  Conversational RAG

create_vectorDB.py creates the vector database. There is no need to run it with the current API Keys (stored in a .env file) as the database has been created already.


Version1.ipynb is the notebook with the version that has the first version of the conversational RAG. Would prefer if further enhancements to the RAG model will be done on a notebook similar to this, and the not the streamlit deployment (for ease of interpretation)


St-Version1.py is the streamlit deployment of Version1.ipynb. To run locally, first run ```python St-Version1.py``` on the terminal. Then, run ```streamlit run St-Version1.py```


data/documents_test.txt is the file the RAG works on. It has all the information of the Data Science Major at Krea.

requirements.txt lists the requirements for this project. It assumes the latest available version of the libraries. Run ```pip install -r requirements.txt``` to install requirements in a virtual environment.


Question.txt lists some sample questions based on data/documents.txt. Can potentially be used in model evaluation (via ```deepeval``` and ```ragas``` packages) through exporting Version1.ipynb as an executable script and integrating ```argpase ``` python package into the script to accept command line arguments.
