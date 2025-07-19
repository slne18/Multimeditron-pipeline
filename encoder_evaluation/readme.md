# Evaluate fine tuned CLIP image encoders

# 0. Install the dependencies with pip install -r requirements_experts.txt

# 1. `similarity_benchmark.py` : Evaluate the encoder ability to associate an image with the most semantically aligned textual description 
For each data sample, the benchmark computes the embedding of the image using the encoder under evaluation, as well as the embedding of its corresponding textual description and those of three random textual distractor descriptions. We then calculate the cosine similarity between the image embedding and each of the four text embeddings. The description with the highest similarity score is selected, and we check whether it matches the correct one. This approach allows us to assess the model’s retrieval capability, specifically its ability to associate an image with the most semantically aligned textual description in a contrastive setting.

