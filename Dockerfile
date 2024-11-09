FROM python:3.10.0-alpine

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --upgrade pip
RUN pip install mkdocs-material==9.5.5 mkdocs-glightbox

WORKDIR /bst236

COPY overrides ./build/overrides

COPY docs ./build/docs
COPY mkdocs.yml mkdocs.yml
RUN mkdocs build -f mkdocs.yml

WORKDIR /bst236/site
EXPOSE 8000
CMD ["python", "-m", "http.server", "8000"]
