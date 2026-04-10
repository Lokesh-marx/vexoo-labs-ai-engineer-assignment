import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch


# You can change this later if needed
MODEL_NAME = "sshleifer/tiny-gpt2"   # lightweight demo model
# For actual target: "meta-llama/Llama-3.2-1B" (if available)


# Step 1: Load dataset
def load_gsm8k_data():
    dataset = load_dataset("openai/gsm8k", "main")

    train_data = dataset["train"].select(range(3000))
    test_data = dataset["test"].select(range(1000))

    return train_data, test_data


# Step 2: Format data
def format_example(example):
    text = f"Question: {example['question']}\nAnswer: {example['answer']}"
    return {"text": text}


# Step 3: Extract final number from answer
def extract_final_number(text):
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text.replace(",", ""))
    return matches[-1] if matches else None


# Step 4: Tokenize data
def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


# Step 5: Main function
def main():
    print("Loading GSM8K dataset...")
    train_data, test_data = load_gsm8k_data()

    print("Formatting dataset...")
    train_data = train_data.map(format_example)
    test_data = test_data.map(format_example)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing dataset...")
    tokenized_train = train_data.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_test = test_data.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask"])
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask"])

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Apply LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # Training args
    training_args = TrainingArguments(
        output_dir="./gsm8k_output",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="no",
        report_to="none"
    )

    # Add labels for causal LM
    def add_labels(example):
        example["labels"] = example["input_ids"]
        return example

    tokenized_train = tokenized_train.map(add_labels)
    tokenized_test = tokenized_test.map(add_labels)

    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train
    )

    print("Starting training...")
    trainer.train()

    print("\nTraining completed successfully!")

    # Simple evaluation (sample 5 examples only for quick demo)
    print("\nRunning simple evaluation on 5 samples...")
    correct = 0
    total = 5

    for i in range(total):
        question = test_data[i]["question"]
        true_answer = test_data[i]["answer"]

        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id
            )

        prediction = tokenizer.decode(output[0], skip_special_tokens=True)

        pred_num = extract_final_number(prediction)
        true_num = extract_final_number(true_answer)

        print(f"\nExample {i+1}")
        print("Question:", question)
        print("Predicted:", pred_num)
        print("Actual:", true_num)

        if pred_num == true_num:
            correct += 1

    accuracy = correct / total
    print(f"\nSample Accuracy on 5 examples: {accuracy:.2f}")


if __name__ == "__main__":
    main()