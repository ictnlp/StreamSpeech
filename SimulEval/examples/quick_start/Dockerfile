FROM python:3.8
RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get install -y \
&& apt-get -y install apt-utils gcc libpq-dev libsndfile-dev
RUN git clone https://github.com/facebookresearch/SimulEval.git
WORKDIR SimulEval
RUN git checkout v1.1.0
RUN pip install -e .
CMD ["simuleval", "--standalone", "--remote-port", "8888", "--agent", "examples/quick_start/first_agent.py.py"]
