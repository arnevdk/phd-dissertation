FROM texlive/texlive
WORKDIR /workdir
RUN apt-get update -y \
	&& apt-get install -y \
	python3-pip \
	&& rm -rf /var/lib/apt/lists/*
COPY code/requirements.txt code/requirements.txt
RUN pip install --break-system-packages --no-cache-dir -r code/requirements.txt
ENV PYTHONPATH=code
