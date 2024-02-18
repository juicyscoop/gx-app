FROM python:3.9-slim
WORKDIR /usr/src/app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
ENV NAME World
CMD ["uvicorn", "src.gx.app:app", "--host", "0.0.0.0", "--reload"]
