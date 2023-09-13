#!/bin/bash
echo "Uploading test data to data.pyrocko.org"
scp data/* pyrocko-www@data.pyrocko.org:/srv/data.pyrocko.org/www/testing/lassie-v2
