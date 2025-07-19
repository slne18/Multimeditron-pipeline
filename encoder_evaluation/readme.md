# Evaluate fine tuned CLIP image encoders

# 0. Install the dependencies with pip install -r requirements_experts.txt

# 1. `similarity_benchmark.py` : Evaluate the encoder ability to associate an image with the most semantically aligned textual description 

For each data sample, the benchmark computes the embedding of the image using the encoder under evaluation, as well as the embedding of its corresponding textual description and those of three random textual distractor descriptions. We then calculate the cosine similarity between the image embedding and each of the four text embeddings. The description with the highest similarity score is selected, and we check whether it matches the correct one. This approach allows us to assess the model’s retrieval capability, specifically its ability to associate an image with the most semantically aligned textual description in a contrastive setting.

Usage : `nohup similarity_benchmark.py > results.out 2> error.out &`

# 2. `anatomical_us_benchmark.py` : evaluate the preservation of anatomically relevant features in the embeddings of the ultrasound images

The benchmark consists of a fully connected feed forward neural network with two hidden layers, which takes as input the image embeddings produced by the encoder under evaluation and classifies it among four different classes: breast, abdomen, thyroid, or others. The network is trained with the embeddings of the evaluated model and after we measure the accuracy of the neural classifier. The dataset used is Radiopaedia.

Usage : `nohup anatomical_us_benchmark.py > results.out 2> error.out &`

