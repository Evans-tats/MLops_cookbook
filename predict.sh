#!/usr/bin/bash

port=8080
echo "using port $port"

curl -X POST http://192.168.100.30:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"weight": 150}'
