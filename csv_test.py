import csv
import time

# Read input CSV
input_filename = 'input.csv'
output_filename = 'output.csv'

with open(input_filename, 'r') as input_file, open(output_filename, 'w', newline='') as output_file:
    csv_reader = csv.DictReader(input_file)
    fieldnames = ['user_question', 'output', 'processing_time']
    csv_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    for row in csv_reader:
        user_question = row['question_column']  # Replace 'question_column' with your actual column name

        # Start timing
        start_time = time.time()

        # Your existing code
        prompt = f"""schema is given and {user_question}"""
        messages = [
            {"role": "system", "content": "You are a SQL expert, given schema and question answer with sql query"},
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_ids))
        results = model.generate_batch([input_tokens], include_prompt_in_result=False, max_length=256, sampling_temperature=0.6, sampling_topp=0.9, end_token=terminators)
        output = tokenizer.decode(results[0].sequences_ids[0])

        # End timing
        end_time = time.time()
        processing_time = end_time - start_time

        # Write to output CSV
        csv_writer.writerow({
            'user_question': user_question, 
            'output': output, 
            'processing_time': f"{processing_time:.2f} seconds"
        })

print(f"Processing complete. Results saved to {output_filename}")
