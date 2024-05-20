#
# see https://python.langchain.com/docs/modules/model_io/models/llms/custom_llm
#
from typing import Any, List, Mapping, Optional
from time import time

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

import oci
from oci.signer import Signer
import requests


class DS_AQUA_LLM(LLM):

    model: str = "odsc-llm" # this is a constant
    debug: bool = False

    max_tokens: int = 300
    temperature: int = 0
    frequency_penalty: int = 1
    top_p: float = 0.75
    top_k: int = 0
    config: Optional[Any] = None
    auth: Optional[Any] = None
    service_endpoint: Optional[str] = None
    compartment_id: Optional[str] = None
    timeout: Optional[int] = 10
    
    """OCI AQUA LLM model.

    To use, you should have the ``oci`` python package installed, and pass 
    named parameters to the constructor.

    Example:
        .. code-block:: python

            compartment_id = "ocid1.compartment.oc1..."
            CONFIG_PROFILE = "my_custom_profile" # or DEFAULT
            config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)
            endpoint = "https://modeldeployment.ap-mumbai-1.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.ap-mumbai-1.amaaaaaap77apcqa4wixlxxjcbzf4ua5kik7mwakgp3nw6fzutajcv2rsdoq/predict"
            llm = DS_AQUA_LLM(
                temperature=0, 
                config=config, 
                # compartment_id=compartment_id, 
                service_endpoint=endpoint
                )


    """

    def __init__(self, **kwargs):
        # print(kwargs)
        super().__init__(**kwargs)
        
        config=self.config
        
        self.auth = Signer(
          tenancy=config['tenancy'],
          user=config['user'],
          fingerprint=config['fingerprint'],
          private_key_file_location=config['key_file']#,
          #pass_phrase=config['pass_phrase']
            )
        
#         auth = Signer(
#           tenancy=config['tenancy'],
#           user=config['user'],
#           fingerprint=config['fingerprint'],
#           private_key_file_location=config['key_file']#,
#           #pass_phrase=config['pass_phrase']
#             )
        
        service_endpoint=self.service_endpoint

    @property
    def _llm_type(self) -> str:
        return "OCI AQUA LLM"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        # calling OCI AQUA
        tStart = time()
        
        body = {
            "model": self.model, # this is a constant
            # "prompt": "what are activation functions?",
            # "prompt": "what is 2+2?",
            "prompt": [prompt],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k
        }

        if self.debug:
            print()
            print("The input prompt is:")
            print(prompt)
            print()
            
        print("Calling OCI AQUA...")
        headers={'Content-Type':'application/json','enable-streaming':'true', 'Accept': 'text/event-stream'}
        response = requests.post(self.service_endpoint, json=body, auth=self.auth, stream=True, headers=headers)

        tEla = time() - tStart

        if self.debug:
            print(f"Elapsed time: {round(tEla, 1)} sec...")
            print()
        # print(f"Full response: {response.json()}")
        return response.json()['choices'][0]['text'].strip()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "config": self.config,
            "auth": self.auth,
            "service_endpoint": self.service_endpoint,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "frequency_penalty": self.frequency_penalty,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
    
    def __repr__(self):
        return f'Model Parameters:\n{self._identifying_params}'
