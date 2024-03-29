#!/bin/bash

set -e # exit script if an error occurs

echo ""
echo "########################################"
echo "Download models, data"
echo "########################################"
echo ""

wget --continue http://data.vision.ee.ethz.ch/alugmayr/SRFlow/datasets.zip
unzip datasets.zip
rm datasets.zip

wget --continue http://data.vision.ee.ethz.ch/alugmayr/SRFlow/pretrained_models.zip
unzip pretrained_models.zip
rm pretrained_models.zip
