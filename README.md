# AKGAN
This is our Pytorch implementation for the paper: "Learning item attributes and user interests for knowledge graph enhanced recommendation"

The proposed model is called AKGAN.

Training AKGAN on Last-FM dataset needs about 13G video memory, and one V100 GPU is ok.

Training AKGAN on Amazon-Book and Alibaba-iFashion datasets needs about 40G video memory, and one A100 GPU is required.

Run the file "main_AKGAN_one.py" to reproduce AKGAN, and the default dataset is Last-FM.

You can also see the training log to check the convergence process of the model.
