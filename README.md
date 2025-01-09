# GenAI-document-classification-extraction

Gen AI Application README
--------------------------
Overview
--------
This application leverages AWS Generative AI services using Amazon Bedrock models (Claude and Nova) through Amazon SageMaker.

Prerequisites
-------------
    1. AWS SageMaker Role Configuration
        Ensure your SageMaker execution role has the following permissions:
        •	Amazon S3 full access
        •	Amazon Bedrock access
        •	IAM permissions for executing the required services
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "s3:*",
                        "bedrock:*"
                    ],
                    "Resource": "*"
                }
            ]
        }

    2. Amazon Bedrock Model Access
        Verify access to the following models in Amazon Bedrock:
        •	Claude 
        •	Nova
        To check model access:
        1.	Go to AWS Console → Bedrock
        2.	Navigate to Model Access
        3.	Enable access to the required models

Installation
------------
1. Clone the Repository
    git clone <repository-url>
    cd <repository-name>

2. Install Dependencies
    Create a virtual environment and install required packages:
    python -m venv venv
    source venv/bin/activate  # For Linux/Mac
    # or
    venv\Scripts\activate  # For Windows
    
    pip install -r requirements.txt
    requirements.tx


Running the Application
-----------------------
Execute the main Python script with required parameters:
python bedrock-realtime-inference.py –opt1 <model-name> --opt2 <input-file>
Parameters
1.	Model Selection (--model):
    o	For Claude model : 'claude-3-5-sonnet','claude-3-sonnet','claude-3-haiku','claude-instant-v1','claude-v2:1', 'claude-v2'
    o	For Nova model: 'nova-pro','nova-lite','nova-micro'
2.	File Name (--filename):
    o	Path to the input file for processing
Example
python bedrock-realtime-inference.py --opt1='claude-3-5-sonnet' --opt2='CWDoc.pdf'
