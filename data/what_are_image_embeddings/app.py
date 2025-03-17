"""This space is taken and modified from https://huggingface.co/spaces/merve/compare_clip_siglip"""
import torch
from transformers import AutoModel, AutoProcessor
import gradio as gr

################################################################################
# Load the models
################################################################################
sg1_ckpt = "google/siglip-so400m-patch14-384"
siglip1_model = AutoModel.from_pretrained(sg1_ckpt, device_map="cpu").eval()
siglip1_processor = AutoProcessor.from_pretrained(sg1_ckpt)

sg2_ckpt = "google/siglip2-so400m-patch14-384"
siglip2_model = AutoModel.from_pretrained(sg2_ckpt, device_map="cpu").eval()
siglip2_processor = AutoProcessor.from_pretrained(sg2_ckpt)


################################################################################
# Utilities
################################################################################
def postprocess_siglip(sg1_probs, sg2_probs, labels):
    sg1_output = {labels[i]: sg1_probs[0][i] for i in range(len(labels))}
    sg2_output = {labels[i]: sg2_probs[0][i] for i in range(len(labels))}
    return sg1_output, sg2_output


def siglip_detector(image, texts):
    sg1_inputs = siglip1_processor(
        text=texts, images=image, return_tensors="pt", padding="max_length", max_length=64
    ).to("cpu")
    sg2_inputs = siglip2_processor(
        text=texts, images=image, return_tensors="pt", padding="max_length", max_length=64
    ).to("cpu")
    
    print("Input texts:", texts)
    print("SigLIP 1 input shapes:", {k: v.shape for k, v in sg1_inputs.items() if hasattr(v, 'shape')})
    print("SigLIP 2 input shapes:", {k: v.shape for k, v in sg2_inputs.items() if hasattr(v, 'shape')})
    
    with torch.no_grad():
        sg1_outputs = siglip1_model(**sg1_inputs)
        sg2_outputs = siglip2_model(**sg2_inputs)
        
        print("\nSigLIP 1 output keys:", sg1_outputs.keys())
        print("SigLIP 2 output keys:", sg2_outputs.keys())
        
        sg1_logits_per_image = sg1_outputs.logits_per_image
        sg2_logits_per_image = sg2_outputs.logits_per_image
        
        print("\nSigLIP 1 logits shape:", sg1_logits_per_image.shape)
        print("SigLIP 1 raw logits:", sg1_logits_per_image)
        print("SigLIP 2 logits shape:", sg2_logits_per_image.shape)
        print("SigLIP 2 raw logits:", sg2_logits_per_image)
        
        sg1_probs = torch.sigmoid(sg1_logits_per_image)
        sg2_probs = torch.sigmoid(sg2_logits_per_image)
        
        print("\nSigLIP 1 probabilities:", sg1_probs)
        print("SigLIP 2 probabilities:", sg2_probs)
        print("-" * 80)
    
    return sg1_probs, sg2_probs


def infer(image, candidate_labels):
    candidate_labels = [label.lstrip(" ") for label in candidate_labels.split(",")]
    sg1_probs, sg2_probs = siglip_detector(image, candidate_labels)
    return postprocess_siglip(sg1_probs, sg2_probs, labels=candidate_labels)


with gr.Blocks() as demo:
    gr.Markdown("# Compare SigLIP 1 and SigLIP 2")
    gr.Markdown(
        "Compare the performance of SigLIP 1 and SigLIP 2 on zero-shot classification in this Space :point_down:"
    )
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil")
            text_input = gr.Textbox(label="Input a list of labels (comma seperated)")
            run_button = gr.Button("Run", visible=True)
        with gr.Column():
            siglip1_output = gr.Label(label="SigLIP 1 Output", num_top_classes=3)
            siglip2_output = gr.Label(label="SigLIP 2 Output", num_top_classes=3)
    examples = [
        ["./baklava.jpg", "dessert on a plate, a serving of baklava, a plate and spoon"],
        ["./cat.jpg", "a cat, two cats, three cats"],
        ["./cat.jpg", "two sleeping cats, two cats playing, three cats laying down"],
    ]
    gr.Examples(
        examples=examples,
        inputs=[image_input, text_input],
        outputs=[siglip1_output, siglip2_output],
        fn=infer,
    )
    run_button.click(fn=infer, inputs=[image_input, text_input], outputs=[siglip1_output, siglip2_output])
demo.launch()
