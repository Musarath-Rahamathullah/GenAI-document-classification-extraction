#load libraries
import boto3
import ast
from anthropic import Anthropic
from botocore.config import Config
import os
import time
import json
import base64
import io
import re
from botocore.exceptions import ClientError
import fitz
from datetime import datetime    
import argparse

# parse the input arguments
parser = argparse.ArgumentParser(description='Script so useful.')
parser.add_argument("--opt1")
parser.add_argument("--opt2")

args = parser.parse_args()

model = args.opt1 #['nova-pro','nova-lite','nova-micro','claude-3-5-sonnet','claude-3-sonnet','claude-3-haiku','claude-instant-v1','claude-v2:1', 'claude-v2']
pdf_filename = args.opt2 # pdf file path


config = Config(
    read_timeout=600, # Read timeout parameter
    retries = dict(
        max_attempts = 10 ## Handle retries
    )
)

# st.set_page_config(initial_sidebar_state="auto")
# Read credentials
with open('config.json','r',encoding='utf-8') as f:
    config_file = json.load(f)
    
BUCKET=config_file["Bucket_Name"]
OUTPUT_TOKEN=config_file["max-output-token"]
REGION=config_file["bedrock-region"]

# pricing info
with open('pricing.json','r',encoding='utf-8') as f:
    pricing_file = json.load(f)

#define the clients
S3 = boto3.client('s3')
bedrock_runtime = boto3.client(service_name='bedrock-runtime',region_name=REGION,config=config)



input_token = 0
output_token = 0
cost = 0


# function to read model output 
def bedrock_streemer(model_id,response, cost, handler):
    stream = response.get('body')
    answer = ""    
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if  chunk:
                chunk_obj = json.loads(chunk.get('bytes').decode())
                
                if "contentBlockDelta" in chunk_obj: 
                    content_block_delta = chunk_obj.get("contentBlockDelta")

                    if "delta" in content_block_delta:                    
                        delta = content_block_delta['delta']
                        if "text" in delta:
                            text=delta['text'] 
                            # st.write(text, end="")                        
                            answer+=str(text)       
                            # handler.markdown(answer.replace("$","USD ").replace("%", " percent"))

                elif "delta" in chunk_obj:                    
                    delta = chunk_obj['delta']
                    if "text" in delta:
                        text=delta['text'] 
                        # st.write(text, end="")                        
                        answer+=str(text)       
                        # handler.markdown(answer.replace("$","USD ").replace("%", " percent"))
                        
                if "amazon-bedrock-invocationMetrics" in chunk_obj:
                    input_token = chunk_obj['amazon-bedrock-invocationMetrics']['inputTokenCount']
                    output_token =chunk_obj['amazon-bedrock-invocationMetrics']['outputTokenCount']

                    if 'claude' in model:
                        pricing=input_token*pricing_file[f"anthropic.{model}"]["input"]+ output_token *pricing_file[f"anthropic.{model}"]["output"]
                    elif 'nova' in model:
                        pricing= input_token *pricing_file[f"amazon.{model}"]["input"]+ output_token *pricing_file[f"amazon.{model}"]["output"]
                    cost+=pricing          
    return answer,cost, input_token, output_token
    
# nova model invoke function
def bedrock_nova_(system_message, prompt,model_id,image_path,cost, handler=None):
    chat_history = []
    content=[]
    if image_path:       
        if not isinstance(image_path, list):
            image_path=[image_path]      
        for img in image_path:
            s3 = boto3.client('s3')
            match = re.match("s3://(.+?)/(.+)", img)
            image_name=os.path.basename(img)
            _,ext=os.path.splitext(image_name)
            if "jpg" in ext.lower(): ext=".jpeg"                        
            if match:
                bucket_name = match.group(1)
                key = match.group(2)    
                obj = s3.get_object(Bucket=bucket_name, Key=key)
                base_64_encoded_data = base64.b64encode(obj['Body'].read())
                base64_string = base_64_encoded_data.decode('utf-8')
            content.extend([
                # {"text":image_name},
                {
              "image": {
                "format": "jpeg",
                "source": {
                    "bytes": base64_string,
                }
              }
            }])
    content.append({
        # "type": "text",
        "text": prompt
            })
    chat_history.append({"role": "user",
            "content": content})
    # print(system_message)
    inf_params = {"max_new_tokens": 4000,"temperature": 0.2}
    prompt = {
        "inferenceConfig": inf_params,
        "system":[{
            "text": system_message
            }
            ],
        "messages": chat_history
    }
    prompt = json.dumps(prompt)
    # print(prompt)
    print(model_id)
    response = bedrock_runtime.invoke_model_with_response_stream(body=prompt, modelId=model_id, accept="application/json", contentType="application/json")
    # print(response)
    answer,cost,input_token, output_token =bedrock_streemer(model_id,response,cost, handler) 
    
    return answer,cost,input_token, output_token


#claude model invoke
def bedrock_claude_(system_message, prompt,model_id,image_path, cost, handler=None):
    content=[]
    chat_history = []
    if image_path:       
        if not isinstance(image_path, list):
            image_path=[image_path]      
        for img in image_path:
            s3 = boto3.client('s3')
            match = re.match("s3://(.+?)/(.+)", img)
            image_name=os.path.basename(img)
            _,ext=os.path.splitext(image_name)
            if "jpg" in ext.lower(): ext=".jpeg"                        
            if match:
                bucket_name = match.group(1)
                key = match.group(2)    
                obj = s3.get_object(Bucket=bucket_name, Key=key)
                base_64_encoded_data = base64.b64encode(obj['Body'].read())
                base64_string = base_64_encoded_data.decode('utf-8')
            
            content.extend([{"type":"text","text":image_name},{
              "type": "image",
              "source": {
                "type": "base64",
                "media_type": f"image/{ext.lower().replace('.','')}",
                "data": base64_string
              }
            }])
    content.append({
        "type": "text",
        "text": prompt
            })
    chat_history.append({"role": "user",
            "content": content})
    # print(system_message)
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4000,
        "temperature": 0.2,
        "system":system_message,
        "messages": chat_history
    }
    
    prompt = json.dumps(prompt)
    # print(prompt)
    # print(model_id)
    response = bedrock_runtime.invoke_model_with_response_stream(body=prompt, modelId=model_id, accept="application/json", contentType="application/json")
    answer,cost,input_token, output_token =bedrock_streemer(model_id,response,cost, handler) 
    return answer, cost,input_token, output_token

# invoke bedrock model with retries, in case of throttling
def _invoke_bedrock_with_retries(chat_template, prompt, model_id, image_path,cost, handler=None):
    max_retries = 10
    backoff_base = 2
    max_backoff = 3  # Maximum backoff time in seconds
    retries = 0

    while True:
        try:
            if 'nova' in model_id:
                response,cost,input_token, output_token = bedrock_nova_(chat_template, prompt, model_id, image_path,cost, handler)
            elif 'claude' in model_id:
                response, cost,input_token, output_token = bedrock_claude_(chat_template, prompt, model_id, image_path,cost, handler)
            return response, cost, input_token, output_token
        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                if retries < max_retries:
                    # Throttling, exponential backoff
                    sleep_time = min(max_backoff, backoff_base ** retries + random.uniform(0, 1))
                    time.sleep(sleep_time)
                    retries += 1
                else:
                    raise e
            elif e.response['Error']['Code'] == 'ModelStreamErrorException':
                if retries < max_retries:
                    # Throttling, exponential backoff
                    sleep_time = min(max_backoff, backoff_base ** retries + random.uniform(0, 1))
                    time.sleep(sleep_time)
                    retries += 1
                else:
                    raise e
            elif e.response['Error']['Code'] == 'EventStreamError':
                if retries < max_retries:
                    # Throttling, exponential backoff
                    sleep_time = min(max_backoff, backoff_base ** retries + random.uniform(0, 1))
                    time.sleep(sleep_time)
                    retries += 1
                else:
                    raise e
            else:
                # Some other API error, rethrow
                raise
                


# combine multiple ojects in json
def combine_json_objects(json_objects):
    combined = {}

    for json_obj in json_objects:
        json_obj = json_obj.replace("```json","").replace("```","")
        json_object = ast.literal_eval(json_obj)

        for section, section_data in json_object.items():
            if section not in combined:
                combined[section] = {}

            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    if isinstance(value, dict):
                        if key not in combined[section]:
                            combined[section][key] = {}
                    
                        if len(value.keys()) != 2:
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, dict):
                                    if sub_key not in combined[section][key]:
                                        combined[section][key][sub_key] = {}
                                    if sub_key in combined[section][key] and "confidence" in combined[section][key][sub_key]:
                                        if sub_value["confidence"] > combined[section][key][sub_key]["confidence"]:
                                            combined[section][key][sub_key] = sub_value

                                    else:
                                        combined[section][key][sub_key] = sub_value
                            
                        elif len(value.keys()) == 2: 
                            if key in combined[section] and "confidence" in combined[section][key]:
                                if value["confidence"] > combined[section][key]["confidence"]:
                                    combined[section][key] = value
                            else:
                                combined[section][key] = value
                        
    return combined

# function to write/append to json
def write_to_json(response,file_name, pageNo, model):
    # Load existing JSON data (or create an empty list if it's a new file)
    try:
        with open('output/data' + file_name + '.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []

    # Create a new object with the response data
    response = {
        "page" : pageNo,
        "output" : response
    }
    # Append the new object to the existing data
    data.append(response)

    # datestamp = str(int(datetime.now().timestamp()))
    # Write the updated data back to the JSON file
    with open('output/' + file_name+'-'+ model + '.json', 'w') as f:
        json.dump(data, f, indent=4)


# convert pdf to multiple image files of each page
def convert_pdf_to_images(pdf_path, output_folder):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    """Converts a PDF file to multiple image files."""
    doc = fitz.open(pdf_path)
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  
        pix = page.get_pixmap()  
        output_file = f"{output_folder}/page_{page_num + 1}.jpeg"
        pix.save(output_file)

    doc.close()
 

# Function takes a user query and a uploaded document and geenrtes a prompt
def query_llm(model,filename):
    """
    Function takes a user query and a uploaded document. Caches documents in S3 is optional
    """  
    doc='I have provided documents and/or images.\n'
    if 'nova' in model:
        model_id='us.amazon.'+model+'-v1:0'
    else:
        model_id='anthropic.'+ model
        if "3-5-sonnet" in model:
            model_id+="-20240620-v1:0"
        elif "sonnet" in model or "haiku" in model:
            model_id+="-20240229-v1:0" if "sonnet" in model else "-20240307-v1:0"

        
    pdf_file = filename #"CWDoc.pdf"#doc_path[0]  # Replace with your PDF file path
    output_dir = pdf_file.split(".pdf")[0] +"/output_images"  # Replace with your desired output directory

    convert_pdf_to_images(pdf_file, output_dir)

    #upload all image files to s3
    for filename in os.listdir(output_dir):
        print("Uploading:", filename)
        S3.upload_file(output_dir+'/'+filename, BUCKET, output_dir+'/'+filename)
    
    with open("prompt/doc_chat.txt","r", encoding="utf-8") as f:
        chat_template=f.read()  

    # entities template 
    with open("prompt/entities.txt","r",encoding="utf-8") as f:
        entity_template=f.read()

    
    json_list = []
    inputTokens = 0
    outputTokens = 0
    cost = 0
    image_files_path = []
    
    paginator = S3.get_paginator('list_objects_v2')
    operation_parameters = {'Bucket': BUCKET,
                            'Prefix': 'output_images'}
    result = paginator.paginate(**operation_parameters)

    # get list of all s3 image file path
    for page in result:
        if "Contents" in page:
            for key in page[ "Contents"]:
                keyString = key[ "Key" ]
                if  keyString.lower().endswith('.jpg') :
                    # print(f"s3://{BUCKET}/{keyString}")
                    image_files_path.append(f"s3://{BUCKET}/{keyString}")

    #iterate over each image file/ document page
    for i,image_path in enumerate(image_files_path): 
        print("Page Number : ", image_path)
        
        prompt=f"""
        You are Claude, an AI assistant specialized in text classification and entity extraction. I will provide you with a piece of image which contains the text, and your task is to classify it into one appropriate category and extract relevant entities from the context and output them in a structured JSON format.  

        <text>
        {doc}
        </text>
        
        <!-- Here are the categories to classify the context -->
        <classification_categories>
        Task Card
        Form 8130
        Form 1
        Logbook Entries
        Other
        </classification_categories>

        <!-- Select the entity based on the above classification category for the JSON response: -->
        <entity_template>
        {entity_template}              
        </entity_template>
                

        <!-- This will tell you how to do your tasks well. -->
        <task_guidance>          
        1. After reading the content provided inside the image, and generate the output as <entity_template></entity_template> format
        2. Do NOT make up answers.
        3. Choose one of the classificaiton options given in <classification_categories></classification_categories>.
        4. If none of the category matches, tag as "Other".
        6. Extract all the entities from <context></context> tags based on the classification category defined in  <entity_template><\entity_template>"
        7. Most important if the classification category in "Other", DO NOT extract any entities. leave the value as empty like, "entitites" : "".
        8. When providing your response:
        - Do not include any preamble or XML tags before and after the json response output.
        - It should not be obvious you are referencing the result.
        - Provide your answer in JSON form. Reply with only the answer in JSON format.
        - json objects should be key-value pair. Do not include type, value in the json object.       
        9. Stricktly do not include any preamble or XML tags before and after the json response output.
        </task_guidance>
        
        <!-- These are your tasks that you must complete -->
        <tasks>
        Profile the content provided to you in the image and double check your work.
        </tasks>   
        """

        
        print("###############################################################")
        # print(each_string)
        
        # # Retrieve past chat history   
        # current_chat,chat_hist=get_chat_history_db(params, CHAT_HISTORY_LENGTH,claude3)

        response, cost,input_token, output_token =_invoke_bedrock_with_retries(chat_template, prompt, model_id, image_path,cost, handler= None)
        
        print(response)
        print("###############################################################")

        pageNo = image_path.split('/')[-1]
        write_to_json(response,pdf_file, pageNo, model)
        
        # Append the new object to the list
        json_list.append(response)
        
        # aggregate all tokens
        inputTokens += input_token
        outputTokens += output_token

    print("Input Tokens:",inputTokens)
    print("Output Tokens:",outputTokens)
    print("Bedrock Cost:", f"${round(cost,2)}")

    # print(json_list)
    json_string = json.dumps(json_list)
    # print(json_string)
    
    return #json.dumps(json_list)


def main(model,pdf_filename):
    start = time.time()
    query_llm(model, pdf_filename)
    minutes = (time.time() - start) / 60
    print(f"Time taken: {minutes:.2f} minutes")
    return 
    
if __name__ == '__main__':
    main(model,pdf_filename)   
    