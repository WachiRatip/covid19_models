FROM python:3.6.9

LABEL basis = "python-pytorch"

WORKDIR /home

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip install jupyter

EXPOSE 8888

VOLUME ./home

COPY ./scr ./

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]