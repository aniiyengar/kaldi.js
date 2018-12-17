X=/Users/aniruddh/git/mrp-www/conference/public/kaldijs
rm $X/*
cp $1-kaldi-worker.js $X
cp $1-kaldi-worker.data $X
cp $1-kaldi-worker.wasm $X
cp kaldi-interop.js $X
