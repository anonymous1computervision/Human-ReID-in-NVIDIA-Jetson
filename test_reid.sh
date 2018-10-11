python3 main_reid.py \
  --embedding_dim=2048 \
  --test_model_dir=models/reid/m74/reid_model.ckpt-200 \
  --test_dir=./data/bounding_box_test/ \
  --query_dir=./data/bounding_box_query/ \
  --train=False \
