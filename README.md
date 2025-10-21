Just a ~simple~ tokeniser based on Google Deepmind's Charformer Paper with the implementation of a gradient based subword tokeniser (GBST)

Current issues - there is no training loop currently so you can see this on the visualisation map as one span seems to overpower importance over other spans
Currently im working on fixing this by implementing said training loop so the tokeniser can better understand what patterns and blocks to give importance to.

Resources used:

Google Deepmind's paper on arXiv
https://arxiv.org/pdf/2106.12672

This wonderful Youtube video by "AI Coffee Break with Letitia"
https://www.youtube.com/watch?v=debgj24BAZE&t=454s
