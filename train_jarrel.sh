bash scripts/tokenizer/train_vq.sh --cloud-save-path /tmp/ctap_tokenizer_32 \
--data-path /tmp/ctap_jpegs \
--image-size 256 \
--vq-model VQ-32 \
--use-encoder-patch \
--codebook-size 32768 \
--results-dir results_tokenizer_image_efficientnet_32 \
--perceptual-weight 0.2 \
--vq-ckpt /home/coder/git/LlamaGen/results_tokenizer_image_efficientnet_32/001-VQ-32/checkpoints/0025000.pt 