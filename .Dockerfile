FROM python:3.11.4-bookworm

WORKDIR /root/code

RUN pip3 install fastapi 
RUN pip3 install uvicorn
RUN pip3 install scikit-learn
RUN pip3 install ipykernel
RUN pip3 install matplotlib
RUN pip3 install seaborn
# RUN pip3 install ppscore
RUN pip3 install scipy
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install Jinja2
RUN pip3 install python-multipart

COPY ./app /root/code

CMD tail -f /dev/null