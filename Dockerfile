FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update
RUN apt-get install python3.10 -y
RUN apt-get install python-is-python3 -y
RUN apt-get install pip -y
RUN apt-get install git -y
RUN apt-get install curl -y
RUN apt-get install wget -y
RUN apt-get install ffmpeg -y

RUN mkdir -p /app/model
RUN mkdir -p /app/sam2
RUN mkdir -p /app/grounding-dino
RUN mkdir -p /app/model/bert-base-uncased

RUN wget "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" -O /app/sam2/sam2.1_hiera_large.pt
RUN wget "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py" -O /app/grounding-dino/GroundingDINO_SwinB.cfg.py
RUN wget "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth" -O /app/grounding-dino/groundingdino_swinb_cogcoor.pth
RUN wget "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/tokenizer_config.json?download=true" -O /app/model/bert-base-uncased/tokenizer_config.json
RUN wget "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/config.json?download=true" -O /app/model/bert-base-uncased/config.json
RUN wget "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt?download=true" -O /app/model/bert-base-uncased/vocab.txt
RUN wget "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/tokenizer.json?download=true" -O /app/model/bert-base-uncased/tokenizer.json
RUN wget "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/model.safetensors?download=true" -O /app/model/bert-base-uncased/model.safetensors

RUN ln -s /app/grounding-dino/groundingdino_swinb_cogcoor.pth /app/model/groundingdino_swinb_cogcoor.pth
RUN ln -s /app/grounding-dino/GroundingDINO_SwinB.cfg.py /app/model/GroundingDINO_SwinB.cfg.py
RUN ln -s /app/sam2/sam2.1_hiera_large.pt /app/model/sam2.1_hiera_large.pt

RUN pip install -q git+https://github.com/pq-yang/MatAnyone2
RUN wget "https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth" -O "/app/model/matanyone.pth"
RUN wget "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth" -O "/app/model/matanyone2.pth"

COPY requirements.txt /app

RUN pip install -r requirements.txt

COPY . /app
RUN chmod +x entrypoint.sh

RUN mkdir -p /jobs
RUN mkdir -p /app/process

ENTRYPOINT ["/app/entrypoint.sh"]
