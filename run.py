import ctranslate2
import transformers
from huggingface_hub import snapshot_download
model_id = "SagarKrishna/Llama-3-8B-Text2SQL_Instruct-ct2-int8_float16"
model_path = snapshot_download(model_id)
model = ctranslate2.Generator(model_path)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
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

print(output)

===================================================================================================================================

import ctranslate2
import transformers

model_id = "Llama-3-8B-Text2SQL_Instruct-ct2-int8_float16"
model = ctranslate2.Generator(model_id, device="cpu")
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)


text = """CREATE TABLE issue_data(
  issueid number,
  username text,
  functionid number,
  issuestate text,
  issuesecstate text,
  issuetype text,
  issuelevel text,
  occurdate number,
  reportdate number,
  impactdata text,
  impactregion text,
  raisedby text,
  raisingfunc text,
  assignedfunc number,
  assignedto text,
  rootcauseapp text,
  rootcausetype text,
  tactsolown text,
  tactsolduedate number,
  startsolown text,
  stratsolduedate number,
  closeddate number,
  escalationdatedqwg number,
  escalationdatedmf number,
  updatedon number,
  updatedby text,
  domainname text,
  domainconcept text,
  level text,
  age number
)
CREATE TABLE rule_metadata (
  dataset_definition_id number,
  datset_query text,
  dataset_name text,
  rule_sql_filter text,
  rule_name text,
  rule_id number,
  rule_display_name text,
  rule_desc text,
  rule_is_active text,
  rule_is_deleted text,
  rule_max_count number,
  rule_provider_id number,
  rule_linked_object_id number,
  rule_calssifier_value_id number,
  rule_sk_ksmm_level number,
  rule_sk_dq_dimension_id number,
  rule_aging_required text,
  rule_is_critical text,
  rule_is_inline_rule text,
  owner text,
  concept text,
  division text
)
CREATE TABLE exception_data(
  rule_id number,
  sk_owner_id number,
  exception_id text,
  exception_category text,
  exception_comments text,
  exception_assignedto text,
  exception_priority text,
  exception_remediationowner text,
  concept text,
  exception_resolutiondate number,
  exception_resolvedby text,
  exception_fixdate number,
  division text,
  exception_state text,
  exception_closurecob number,
  exception_exceptionitemid text,
  exception_subcategory text,
  exception_currentage number,
  owner text,
  exception_cobdate number
);

--Using valid SQLite,answer the following questions for the tables provided above.

--show rule ids with open exception state?(generate only 1 sql query and dont assume any other conditions in where clause)

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

print(output)

