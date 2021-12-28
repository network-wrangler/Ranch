#to build the shst image, run the following line
#docker build -t shst .
#example commands to run the following lines in cmd
#docker run -it --rm -v /local/path:/usr/node/ shst:latest /bin/bash
#docker run -it --rm -v /local/path:/usr/node/ shst:latest shst extract usr/node/boundary_ii.geojson --out=usr/node/ii.geojson --metadata --tile-hierarchy=8 --tiles
#docker run -it --rm -v /local/path:/usr/node/ shst:latest shst match /usr/node/point_1.in.geojson --out=/usr/node/out.geojson --tile-hierarchy=8

FROM node:12-bullseye-slim

#if the user run into local certificate issues, include the following two lines
#cacert.pem here is the example certificate file
#ADD cacert.pem /usr/local/share/ca-certificates/cacert.pem
#ENV NODE_EXTRA_CA_CERTS=/usr/local/share/ca-certificates/cacert.pem

ENV NPM_CONFIG_PREFIX=/home/node/.npm-global
ENV PATH=$PATH:/home/node/.npm-global/bin

USER node
RUN npm install -g sharedstreets
