FROM python:3.11.4-bookworm

WORKDIR /root/code

RUN pip3 install fastapi 
RUN pip3 install uvicorn
RUN pip3 install scikit-learn==1.3.2
RUN pip3 install ipykernel
RUN pip3 install matplotlib
RUN pip3 install seaborn
# RUN pip3 install ppscore
RUN pip3 install scipy
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install Jinja2
RUN pip3 install python-multipart
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install mlflow 


# Testing Modules
RUN pip3 install fastapi[testing]
RUN pip3 install pytest
RUN pip3 install pytest-depends

COPY . /root/code



CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
