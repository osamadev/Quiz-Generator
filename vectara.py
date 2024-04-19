import json
import logging
import mimetypes
import os
from typing import Tuple

import requests
from authlib.integrations.requests_client import OAuth2Session
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Indexing:
    def __init__(self):
        self.auth_url = os.getenv('AUTH_URL')
        self.app_client_id = os.getenv('APP_CLIENT_ID')
        self.app_client_secret = os.getenv('APP_CLIENT_SECRET')
        self.jwt_token = self._get_jwt_token()
        self.customer_id = os.getenv('CUSTOMER_ID')
        self.api_key = os.getenv('API_KEY')
        self.corpus_id = os.getenv('CORPUS_ID')

    def _get_jwt_token(self) -> str:
        """Connect to the server and get a JWT token."""
        token_endpoint = f"{self.auth_url}/oauth2/token"
        session = OAuth2Session(self.app_client_id, self.app_client_secret, scope="")
        token = session.fetch_token(token_endpoint, grant_type="client_credentials")
        return token["access_token"]
    
    def upload_file(self, customer_id: int, corpus_id: int, idx_address: str, uploaded_file, file_title: str) -> Tuple[requests.Response, bool]:
            """Uploads a file to the corpus."""
            # Determine the MIME type based on the file extension
            extension_to_mime_type = {
                '.txt': 'text/plain',
                '.pdf': 'application/pdf',
                '.doc': 'application/msword',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                '.xlsx' : 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                # ... add more mappings as needed
            }
            file_extension = os.path.splitext(uploaded_file.name)[-1]
            mime_type = extension_to_mime_type.get(file_extension, 'application/octet-stream')

            #mime_type = mimetypes.guess_type(uploaded_file.name)[0] or 'application/octet-stream'

            post_headers = {
                "Authorization": f"Bearer {self.jwt_token}"
            }
            
            try:
                file = uploaded_file.read()  
                files = {"file": (file_title, file, mime_type)}
                response = requests.post(
                    f"https://{idx_address}/v1/upload?c={customer_id}&o={corpus_id}",
                    files=files,
                    headers=post_headers
                )
            
                if response.status_code != 200:
                    logging.error("REST upload failed with code %d, reason %s, text %s",
                                response.status_code,
                                response.reason,
                                response.text)
                    return response, False
                return response, True
            except Exception as e:
                logging.error("An error occurred while uploading the file: %s", str(e))
                return None, False
    
    def get_post_headers(self) -> dict:
        """Returns headers that should be attached to each post request."""
        return {
            "x-api-key": self.api_key,
            "customer-id": self.customer_id,
            "Content-Type": "application/json",
        }

    def index_doc(self, session, doc: dict) -> str:
        req = {
            "customerId": self.customer_id,
            "corpusId": self.corpus_id,
            "document": doc
        }

        response = session.post(
            headers=self.get_post_headers(),
            url="https://api.vectara.io/v1/index",
            data=json.dumps(req),
            timeout=250,
            verify=True,
        )

        status_code = response.status_code
        result = response.json()
        
        status_str = result["status"]["code"] if "status" in result else None
        if status_code == 409 or status_str and (status_str == "ALREADY_EXISTS"):
            return "E_ALREADY_EXISTS"
        elif status_str and (status_str == "FORBIDDEN"):
            return "E_NO_PERMISSIONS"
        else:
            return "E_SUCCEEDED"


class Searching:
    def __init__(self):
        self.customer_id = os.getenv('CUSTOMER_ID')
        self.api_key = os.getenv('API_KEY')

    def send_query(self, corpus_id, query_text, num_results, summarizer_prompt_name, response_lang, max_summarized_results):
        api_key_header = {
            "customer-id": self.customer_id,
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        data_dict = {
            "query": [
                {
                    "query": query_text,
                    "num_results": num_results,
                    "corpus_key": [{"customer_id": self.customer_id, "corpus_id": corpus_id}],
                    'summary': [
                        {
                            'summarizerPromptName': summarizer_prompt_name,
                            'responseLang': response_lang,
                            'maxSummarizedResults': max_summarized_results
                        }
                    ]
                }
            ]
        }

        payload = json.dumps(data_dict)

        response = requests.post(
            "https://api.vectara.io/v1/query",
            data=payload,
            verify=True,
            headers=api_key_header
        )

        if response.status_code == 200:
            print("Request was successful!")
            data = response.json()
            texts = [item['text'] for item in data['responseSet'][0]['response'] if 'text' in item]
            return texts
        else:
            print("Request failed with status code:", response.status_code)
            print("Response:", response.text)
            return None
        
