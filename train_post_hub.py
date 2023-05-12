from transformers import DistilBertForSequenceClassification

OUTPUT_DIR = "experiments"
pt_model = DistilBertForSequenceClassification.from_pretrained(OUTPUT_DIR, from_tf=True)
pt_model.save_pretrained(OUTPUT_DIR)
pt_model.push_to_hub("devFacebooks/trainai")