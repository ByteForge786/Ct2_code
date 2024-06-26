import streamlit as st
import ctranslate2
import transformers

def generate_sql_query(schema, question):
    model_id = "Llama-3-8B-Text2SQL_Instruct-ct2-int8_float16"
    model = ctranslate2.Generator(model_id, device="cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    text = f"""{schema}
--Using valid SQLite,answer the following questions for the tables provided above.
--{question}
answer:"""

    messages = [
        {"role": "system", "content": "You are SQL Expert. Given Schema and Question, answer with sql query"},
        {"role": "user", "content": text},
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
    
    return output

def main():
    st.title("SQL Query Generator")

    # Text area for schema input
    schema_input = st.text_area("Enter your table schema:", height=200)

    # Text input for question
    question_input = st.text_input("Enter your question:")

    if st.button("Generate SQL Query"):
        if schema_input and question_input:
            with st.spinner("Generating SQL query..."):
                generated_query = generate_sql_query(schema_input, question_input)
            st.subheader("Generated SQL Query:")
            st.code(generated_query, language="sql")
        else:
            st.warning("Please enter both the schema and the question.")

if __name__ == "__main__":
    main()
