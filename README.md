## Basic LLM calling with LangChain

### Installation:

run the following command to install the required packages:

```bash
python -m venv venv

# for windows:
.\venv\Scripts\activate
# or for macos/linux:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install jupyter ipykernel
python -m ipykernel install --user --name=llm_course_kernel --display-name "LLM Course Kernel"\

jupyter notebook
```